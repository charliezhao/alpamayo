# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import inspect
import logging
from typing import Any

import einops
import hydra.utils as hyu
import numpy as np
import torch
from transformers import AutoConfig, AutoModel, StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from alpamayo_r1.action_space import ActionSpace
from alpamayo_r1.models.base_model import ReasoningVLA
from alpamayo_r1.config import AlpamayoR1Config
from alpamayo_r1.diffusion.base import BaseDiffusion
from alpamayo_r1.models.token_utils import (
    StopAfterEOS,
    extract_text_tokens,
    replace_padding_after_eos,
    to_special_token,
)

logger = logging.getLogger(__name__)


class NvtxRange:
    """
    Safe NVTX range context manager.

    - No-op if CUDA/NVTX isn't available.
    - Prevents unbalanced push/pop on exceptions.
    """
    def __init__(self, name: str):
        self.name = name
        self._enabled = bool(
            torch.cuda.is_available()
            and hasattr(torch.cuda, "nvtx")
            and hasattr(torch.cuda.nvtx, "range_push")
            and hasattr(torch.cuda.nvtx, "range_pop")
        )

    def __enter__(self):
        if self._enabled:
            torch.cuda.nvtx.range_push(self.name)
        return self

    def __exit__(self, *args):
        if self._enabled:
            torch.cuda.nvtx.range_pop()


class ExpertLogitsProcessor(LogitsProcessor):
    """Masks out the logits for discrete trajectory tokens."""

    def __init__(self, traj_token_offset: int, traj_vocab_size: int):
        """Initialize the ExpertLogitsProcessor.

        Args:
            traj_token_offset: The offset of the trajectory tokens.
            traj_vocab_size: The vocabulary size of the trajectory tokens.
        """
        super().__init__()
        self.traj_token_offset = traj_token_offset
        self.traj_vocab_size = traj_vocab_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Call the ExpertLogitsProcessor to mask out the logits for discrete trajectory tokens.

        The discrete trajectory tokens are not used for the expert model thus masking them out for
        better CoC generation.

        Args:
            input_ids: The input IDs.
            scores: The scores.

        Returns:
            torch.FloatTensor: The modified scores tensor with trajectory tokens masked out (set to -inf).
        """
        # Directly assign -inf to the trajectory token positions in the scores tensor
        scores[:, self.traj_token_offset : self.traj_token_offset + self.traj_vocab_size] = float('-inf')
        return scores


class AlpamayoR1(ReasoningVLA):
    """Expert model for reasoning VLA."""

    config_class: type[AlpamayoR1Config] = AlpamayoR1Config
    base_model_prefix = "vlm"

    def __init__(
        self,
        config: AlpamayoR1Config,
        pretrained_modules: dict[str, torch.nn.Module] | None = None,
        original_vocab_size: int | None = None,
    ):
        super().__init__(config, pretrained_modules, original_vocab_size, print_param_count=False)

        # we only need the text config for the expert model
        expert_config = copy.deepcopy(self.vlm.config.text_config)
        if config.expert_cfg is not None:
            for key, value in config.expert_cfg.items():
                setattr(expert_config, key, value)
        self.expert = AutoModel.from_config(expert_config)
        # we don't need the embed_tokens of the expert model
        del self.expert.embed_tokens

        self.action_space: ActionSpace = hyu.instantiate(config.action_space_cfg)
        self.diffusion: BaseDiffusion = hyu.instantiate(
            config.diffusion_cfg,
            x_dims=self.action_space.get_action_space_dims(),
        )

        self.action_in_proj = hyu.instantiate(
            config.action_in_proj_cfg,
            in_dims=self.action_space.get_action_space_dims(),
            out_dim=expert_config.hidden_size,
        )
        self.action_out_proj = hyu.instantiate(
            config.action_out_proj_cfg,
            in_features=expert_config.hidden_size,
            out_features=self.action_space.get_action_space_dims()[-1],
        )

        # Convert action-related modules to the same dtype as expert
        expert_dtype = self.expert.dtype
        if self.config.keep_same_dtype:
            self.diffusion = self.diffusion.to(dtype=expert_dtype)
            self.action_in_proj = self.action_in_proj.to(dtype=expert_dtype)
            self.action_out_proj = self.action_out_proj.to(dtype=expert_dtype)

        self.post_init()

    def sample_trajectories_from_data_with_vlm_rollout(
        self,
        data: dict[str, Any],
        top_p: float = 0.98,
        top_k: int | None = None,
        temperature: float = 0.6,
        num_traj_samples: int = 6,
        num_traj_sets: int = 1,
        diffusion_kwargs: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        with NvtxRange("0_Total_Alpamayo_Inference"):
            # Global setup
            n_samples_total = num_traj_samples * num_traj_sets
            ego_history_xyz = data["ego_history_xyz"]
            ego_history_rot = data["ego_history_rot"]
            B = ego_history_xyz.shape[0]
            tokenized_data = data["tokenized_data"]
            
            # --- STEP 1: VISION ENCODER (Target: 3.43ms) ---
            # This block now exclusively handles sensor fusion and vision tokenization
            with NvtxRange("1_1_Vision_Encoder_Trigger"):
                input_ids = self.fuse_traj_tokens(tokenized_data.pop("input_ids"), {
                    "ego_history_xyz": ego_history_xyz, 
                    "ego_history_rot": ego_history_rot
                })
                device = input_ids.device
                # Force GPU to finish vision kernels so the marker boundary is visible
                torch.cuda.synchronize()

            # --- STEP 2: PREFILLING (Target: 16.54ms) ---
            with NvtxRange("1_2_Vision_Encoder_Setup"):
                gen_config = copy.deepcopy(self.vlm.generation_config)
                gen_config.update(
                    top_p=top_p, temperature=temperature, do_sample=True,
                    num_return_sequences=num_traj_samples,
                    max_new_tokens=1, # Trigger prompt prefill pass
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    use_cache=True
                )
                
                prefill_output = self.vlm.generate(
                    input_ids=input_ids, 
                    generation_config=gen_config,
                    **tokenized_data
                )
                torch.cuda.synchronize()

            # --- STEP 3: REASONING DECODING (Target: 70ms) ---
            with NvtxRange("3_Reasoning_Decoding"):
                eos_id = self.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
                stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_id)])
                logits_proc = LogitsProcessorList([ExpertLogitsProcessor(self.config.traj_token_start_idx, self.config.traj_vocab_size)])

                gen_config.max_new_tokens = kwargs.get("max_generation_length", self.config.tokens_per_future_traj)
                
                vlm_outputs = self.vlm.generate(
                    input_ids=prefill_output.sequences, 
                    past_key_values=prefill_output.past_key_values,
                    generation_config=gen_config,
                    stopping_criteria=stopping_criteria, 
                    logits_processor=logits_proc,
                )
                torch.cuda.synchronize()

            # --- STEP 4: TRAJECTORY DECODING (Target: 8.75ms) ---
            with NvtxRange("4_Trajectory_Decoding"):
                vlm_tokens = replace_padding_after_eos(vlm_outputs.sequences, eos_id, self.tokenizer.pad_token_id)
                prompt_cache = vlm_outputs.past_key_values
                prefill_seq_len = prompt_cache.get_seq_length()
                b_star = vlm_tokens.shape[0]
                
                traj_mask = vlm_tokens == eos_id
                offset = torch.where(traj_mask.any(dim=1), traj_mask.int().argmax(dim=1), vlm_tokens.shape[1] - 1) + 1
                n_diff_tokens = self.action_space.get_action_space_dims()[0]
                
                pos_ids = einops.repeat(torch.arange(n_diff_tokens, device=device), "l -> 3 b l", b=b_star).clone()
                pos_ids += (self.vlm.model.rope_deltas + offset[:, None]).to(device)
                
                attn_mask = torch.zeros((b_star, 1, n_diff_tokens, prefill_seq_len + n_diff_tokens), device=device)
                for i in range(b_star):
                    attn_mask[i, :, :, offset[i] : -n_diff_tokens] = torch.finfo(torch.float32).min

                def step_fn(x, t):
                    with NvtxRange("diffusion_step_inner"):
                        x_emb = self.action_in_proj(x, t)
                        if x_emb.dim() == 2:
                            x_emb = x_emb.view(b_star, n_diff_tokens, -1)
                        expert_out = self.expert(
                            inputs_embeds=x_emb, position_ids=pos_ids, past_key_values=prompt_cache,
                            attention_mask=attn_mask, use_cache=True
                        )
                        prompt_cache.crop(prefill_seq_len)
                        return self.action_out_proj(expert_out.last_hidden_state[:, -n_diff_tokens:]).view(-1, *self.action_space.get_action_space_dims())

                diffusion_kwargs = {**(diffusion_kwargs or {}), "num_steps": 5}
                sampled_action = self.diffusion.sample(batch_size=B * n_samples_total, step_fn=step_fn, device=device, **diffusion_kwargs)
                torch.cuda.synchronize()

            """
            # Final physical mapping
            p_xyz, p_rot = self.action_space.action_to_traj(sampled_action, 
                einops.repeat(ego_history_xyz[:, -1], "b ... -> (b n) ...", n=n_samples_total), 
                einops.repeat(ego_history_rot[:, -1], "b ... -> (b n) ...", n=n_samples_total))
            
            return einops.rearrange(p_xyz, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples), \
                   einops.rearrange(p_rot, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples)
            """
            
            # Map back to original variable names for return logic
            hist_xyz = einops.repeat(ego_history_xyz[:, -1], "b ... -> (b n) ...", n=n_samples_total)
            hist_rot = einops.repeat(ego_history_rot[:, -1], "b ... -> (b n) ...", n=n_samples_total)
            p_xyz, p_rot = self.action_space.action_to_traj(sampled_action, hist_xyz, hist_rot)
            
            pred_xyz = einops.rearrange(p_xyz, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples)
            pred_rot = einops.rearrange(p_rot, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples)

            # --- ORIGINAL RETURN LOGIC ---
            if kwargs.get("return_extra", False):
                extra = extract_text_tokens(self.tokenizer, vlm_outputs.sequences)
                for text_tokens in extra.keys():
                    extra[text_tokens] = np.array(extra[text_tokens]).reshape(
                        [input_ids.shape[0], num_traj_sets, num_traj_samples]
                    )
                return pred_xyz, pred_rot, extra
            
            return pred_xyz, pred_rot            

AutoConfig.register("alpamayo_r1", AlpamayoR1Config)
AutoModel.register(AlpamayoR1Config, AlpamayoR1)

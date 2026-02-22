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
        scores[:, self.traj_token_offset : self.traj_token_offset + self.traj_vocab_size] = float("-inf")
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

    def sample_trajectories_from_data_with_vlm_rollout__backup( #charlie-change
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample trajectories from the data with VLM rollout.

        Args:
            data: The input data.
            top_p: The top-p value for sampling.
            top_k: The top-k value for sampling.
            temperature: The temperature for sampling.
            num_traj_samples: The number of trajectory samples.
            num_traj_sets: The number of trajectory sets.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            pred_xyz: The predicted xyz.
            pred_rot: The predicted rotation.
            logprob: The log probability.
        """
        with NvtxRange("0_Total_Inference"): # Root range
            n_samples_total = num_traj_samples * num_traj_sets
            ego_history_xyz = data["ego_history_xyz"]
            ego_history_rot = data["ego_history_rot"]
            B, n_traj_group, _, _ = ego_history_xyz.shape
            assert n_traj_group == 1, "Only one trajectory group is supported for inference."
            tokenized_data = data["tokenized_data"]
            input_ids = tokenized_data.pop("input_ids")
            traj_data_vlm = {
                "ego_history_xyz": ego_history_xyz,
                "ego_history_rot": ego_history_rot,
            }
            input_ids = self.fuse_traj_tokens(input_ids, traj_data_vlm)
            device = input_ids.device

            # -------------------------
            # 1) VLM autoregressive generation
            # -------------------------
            with NvtxRange("vlm_generation_total"):
                max_generation_length = kwargs.get(
                    "max_generation_length", self.config.tokens_per_future_traj
                )

                # IMPORTANT: avoid mutating self.vlm.generation_config in-place
                generation_config = copy.deepcopy(self.vlm.generation_config)
                generation_config.top_p = top_p
                generation_config.temperature = temperature
                generation_config.do_sample = True
                generation_config.num_return_sequences = num_traj_samples
                generation_config.max_new_tokens = max_generation_length
                generation_config.output_logits = True
                generation_config.return_dict_in_generate = True
                generation_config.top_k = top_k
                generation_config.pad_token_id = self.tokenizer.pad_token_id

                # custom stopping: stop after <traj_future_start> + one more token
                eos_token_id = self.tokenizer.convert_tokens_to_ids(
                    to_special_token("traj_future_start")
                )
                stopping_criteria = StoppingCriteriaList(
                    [StopAfterEOS(eos_token_id=eos_token_id)]
                )
                logits_processor = LogitsProcessorList(
                    [
                        ExpertLogitsProcessor(
                            traj_token_offset=self.config.traj_token_start_idx,
                            traj_vocab_size=self.config.traj_vocab_size,
                        )
                    ]
                )

                with NvtxRange("vision_prefill_decode"):
                    vlm_outputs = self.vlm.generate(
                        input_ids=input_ids,
                        generation_config=generation_config,
                        stopping_criteria=stopping_criteria,
                        logits_processor=logits_processor,
                        **tokenized_data,
                    )

            # rope deltas used later for position ids
            vlm_outputs.rope_deltas = self.vlm.model.rope_deltas

            # -------------------------
            # CPU-side post-processing to build expert inputs
            # (keep OUTSIDE trajectory timing)
            # -------------------------
            vlm_outputs.sequences = replace_padding_after_eos(
                token_ids=vlm_outputs.sequences,
                eos_token_id=eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            prompt_cache = vlm_outputs.past_key_values
            prefill_seq_len = prompt_cache.get_seq_length()

            b_star = vlm_outputs.sequences.shape[0]
            traj_future_start_mask = vlm_outputs.sequences == eos_token_id
            has_traj_future_start = traj_future_start_mask.any(dim=1)
            for i in range(b_star):
                if not has_traj_future_start[i]:
                    logger.warning(
                        f"No <traj_future_start> token found in the generated sequences for sequence {i}"
                    )

            traj_future_start_positions = traj_future_start_mask.int().argmax(dim=1)
            last_token_positions = torch.full(
                (b_star,), vlm_outputs.sequences.shape[1] - 1, device=device
            )
            valid_token_pos_id = torch.where(
                has_traj_future_start, traj_future_start_positions, last_token_positions
            )
            offset = valid_token_pos_id + 1

            n_diffusion_tokens = self.action_space.get_action_space_dims()[0]
            position_ids = torch.arange(n_diffusion_tokens, device=device)
            position_ids = einops.repeat(position_ids, "l -> 3 b l", b=b_star).clone()
            delta = vlm_outputs.rope_deltas + offset[:, None]
            position_ids += delta.to(position_ids.device)

            attention_mask = torch.zeros(
                (b_star, 1, n_diffusion_tokens, prompt_cache.get_seq_length() + n_diffusion_tokens),
                dtype=torch.float32,
                device=device,
            )
            for i in range(b_star):
                attention_mask[i, :, :, offset[i] : -n_diffusion_tokens] = torch.finfo(
                    attention_mask.dtype
                ).min

            forward_kwargs = {}
            if self.config.expert_non_causal_attention:
                forward_kwargs["is_causal"] = False

            # -------------------------
            # 2) Define denoising step (called repeatedly inside diffusion sampler)
            # -------------------------
            def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                b_star_local = x.shape[0]

                future_token_embeds = self.action_in_proj(x, t)
                if future_token_embeds.dim() == 2:
                    future_token_embeds = future_token_embeds.view(
                        b_star_local, n_diffusion_tokens, -1
                    )

                expert_out_base = self.expert(
                    inputs_embeds=future_token_embeds,
                    position_ids=position_ids,
                    past_key_values=prompt_cache,
                    attention_mask=attention_mask,
                    use_cache=True,
                    **forward_kwargs,
                )

                # crop prompt cache to remove newly added tokens
                prompt_cache.crop(prefill_seq_len)

                last_hidden = expert_out_base.last_hidden_state
                last_hidden = last_hidden[:, -n_diffusion_tokens:]

                pred = self.action_out_proj(last_hidden).view(
                    -1, *self.action_space.get_action_space_dims()
                )
                return pred

            # -------------------------
            # 3) Diffusion sampling in action space
            # -------------------------
            total_batch = B * n_samples_total
            if diffusion_kwargs is None:
                diffusion_kwargs = {}
            else:
                # avoid mutating caller-provided dict
                diffusion_kwargs = dict(diffusion_kwargs)

            # Force paper setting: 5 steps, but only set a parameter name the sampler supports.
            # This prevents errors from unknown kwargs.
            forced_steps = 5
            sig = None
            try:
                sig = inspect.signature(self.diffusion.sample)
            except Exception:
                sig = None

            if sig is not None:
                # choose the first supported parameter name
                for k in ("num_steps", "steps", "num_inference_steps", "num_diffusion_steps"):
                    if k in sig.parameters:
                        diffusion_kwargs[k] = forced_steps
                        break
            else:
                # fallback (most common)
                diffusion_kwargs["num_steps"] = forced_steps

            with NvtxRange("trajectory_decode"):
                sampled_action = self.diffusion.sample(
                    batch_size=total_batch,
                    step_fn=step_fn,
                    device=device,
                    return_all_steps=False,
                    **diffusion_kwargs,
                )

            # -------------------------
            # 4) Convert actions to trajectories + reshape
            # -------------------------
            hist_xyz_rep = einops.repeat(
                ego_history_xyz[:, -1], "b ... -> (b n) ...", n=n_samples_total
            )
            hist_rot_rep = einops.repeat(
                ego_history_rot[:, -1], "b ... -> (b n) ...", n=n_samples_total
            )

            pred_xyz, pred_rot = self.action_space.action_to_traj(
                sampled_action, hist_xyz_rep, hist_rot_rep
            )

            pred_xyz = einops.rearrange(
                pred_xyz, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
            )
            pred_rot = einops.rearrange(
                pred_rot, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
            )

            # return the text tokens generated by the VLM
            if kwargs.get("return_extra", False):
                extra = extract_text_tokens(self.tokenizer, vlm_outputs.sequences)
                for text_tokens in extra.keys():
                    extra[text_tokens] = np.array(extra[text_tokens]).reshape(
                        [input_ids.shape[0], num_traj_sets, num_traj_samples]
                    )
                return pred_xyz, pred_rot, extra

            return pred_xyz, pred_rot

    def sample_trajectories_from_data_with_vlm_rollout__3steps(
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
            # Variables needed across multiple steps defined in root scope
            n_samples_total = num_traj_samples * num_traj_sets
            ego_history_xyz = data["ego_history_xyz"]
            ego_history_rot = data["ego_history_rot"]
            B = ego_history_xyz.shape[0] # Defined here for global access
            tokenized_data = data["tokenized_data"]
            
            # --- STEP 1: VISION ENCODER (Target: ~3.43ms) ---
            with NvtxRange("1_1_Vision_Encoder_Trigger"):
                input_ids = self.fuse_traj_tokens(tokenized_data.pop("input_ids"), {
                    "ego_history_xyz": ego_history_xyz, 
                    "ego_history_rot": ego_history_rot
                })
                device = input_ids.device
                torch.cuda.synchronize() # Force marker boundary

            # --- STEP 2: PREFILLING (Target: ~16.54ms) ---
            with NvtxRange("1_2_Vision_Encoder_Setup"):
                gen_config = copy.deepcopy(self.vlm.generation_config)
                gen_config.update(
                    top_p=top_p, temperature=temperature, do_sample=True,
                    num_return_sequences=num_traj_samples,
                    max_new_tokens=1, # Generate 1 token to trigger prefill
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    use_cache=True
                )
                
                # First pass to process the vision/prompt tokens and populate KV-cache
                prefill_output = self.vlm.generate(
                    input_ids=input_ids, 
                    generation_config=gen_config,
                    **tokenized_data
                )
                torch.cuda.synchronize()

            # --- STEP 3: REASONING DECODING (Target: ~70ms) ---
            with NvtxRange("3_Reasoning_Decoding"):
                eos_id = self.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
                stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_id)])
                logits_proc = LogitsProcessorList([ExpertLogitsProcessor(self.config.traj_token_start_idx, self.config.traj_vocab_size)])

                # Resume decoding using the KV-cache from Step 2
                gen_config.max_new_tokens = kwargs.get("max_generation_length", self.config.tokens_per_future_traj)
                
                outputs = self.vlm.generate(
                    input_ids=prefill_output.sequences, 
                    past_key_values=prefill_output.past_key_values,
                    generation_config=gen_config,
                    stopping_criteria=stopping_criteria, 
                    logits_processor=logits_proc,
                )
                torch.cuda.synchronize()

            # --- STEP 4: TRAJECTORY DECODING (Target: ~8.75ms) ---
            with NvtxRange("4_Trajectory_Decoding"):
                vlm_tokens = replace_padding_after_eos(outputs.sequences, eos_id, self.tokenizer.pad_token_id)
                prompt_cache = outputs.past_key_values
                prefill_seq_len = prompt_cache.get_seq_length()
                b_star = vlm_tokens.shape[0]
                
                traj_mask = vlm_tokens == eos_id
                offset = torch.where(traj_mask.any(dim=1), traj_mask.int().argmax(dim=1), vlm_tokens.shape[1] - 1) + 1
                n_diff_tokens = self.action_space.get_action_space_dims()[0]
                
                # Position and attention setup for Blackwell
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

            # Mapping diffusion result back to physical trajectory space
            p_xyz, p_rot = self.action_space.action_to_traj(sampled_action, 
                einops.repeat(ego_history_xyz[:, -1], "b ... -> (b n) ...", n=n_samples_total), 
                einops.repeat(ego_history_rot[:, -1], "b ... -> (b n) ...", n=n_samples_total))
            
            return einops.rearrange(p_xyz, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples), \
                   einops.rearrange(p_rot, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples)    


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
                
                outputs = self.vlm.generate(
                    input_ids=prefill_output.sequences, 
                    past_key_values=prefill_output.past_key_values,
                    generation_config=gen_config,
                    stopping_criteria=stopping_criteria, 
                    logits_processor=logits_proc,
                )
                torch.cuda.synchronize()

            # --- STEP 4: TRAJECTORY DECODING (Target: 8.75ms) ---
            with NvtxRange("4_Trajectory_Decoding"):
                vlm_tokens = replace_padding_after_eos(outputs.sequences, eos_id, self.tokenizer.pad_token_id)
                prompt_cache = outputs.past_key_values
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

            # Final physical mapping
            p_xyz, p_rot = self.action_space.action_to_traj(sampled_action, 
                einops.repeat(ego_history_xyz[:, -1], "b ... -> (b n) ...", n=n_samples_total), 
                einops.repeat(ego_history_rot[:, -1], "b ... -> (b n) ...", n=n_samples_total))
            
            return einops.rearrange(p_xyz, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples), \
                   einops.rearrange(p_rot, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples)

AutoConfig.register("alpamayo_r1", AlpamayoR1Config)
AutoModel.register(AlpamayoR1Config, AlpamayoR1)


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

# End-to-end example script for the inference pipeline:
# This script loads a dataset, runs inference, and computes the minADE.
# It can be used to test the inference pipeline.

import json
import os
import time
from contextlib import contextmanager

import numpy as np
import psutil
import torch

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper


PROC = psutil.Process(os.getpid())
profile_stats = []


@contextmanager
def stage_timer(name, stats_list):
    rss0 = PROC.memory_info().rss
    cpu_times0 = PROC.cpu_times()
    t0 = time.perf_counter()

    yield

    t1 = time.perf_counter()
    cpu_times1 = PROC.cpu_times()
    rss1 = PROC.memory_info().rss

    wall_s = t1 - t0
    cpu_s = (cpu_times1.user - cpu_times0.user) + (cpu_times1.system - cpu_times0.system)
    cpu_pct = (cpu_s / wall_s * 100.0) if wall_s > 0 else 0.0

    row = {
        "stage": name,
        "wall_ms": wall_s * 1000.0,
        "cpu_ms": cpu_s * 1000.0,
        "cpu_pct": cpu_pct,
        "rss_delta_mb": (rss1 - rss0) / (1024 ** 2),
    }
    stats_list.append(row)
    print(
        f"[PROFILE] {name:24s} "
        f"wall={row['wall_ms']:9.2f} ms  "
        f"cpu={row['cpu_ms']:9.2f} ms  "
        f"cpu%={row['cpu_pct']:7.1f}  "
        f"rss_delta={row['rss_delta_mb']:8.2f} MB"
    )


# Example clip ID
clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
print(f"Loading dataset for clip_id: {clip_id}...")

with stage_timer("load_dataset_sample", profile_stats):
    data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)

print("Dataset loaded.")

print("=== AFTER DATA LOAD ===")
print("image_frames.shape =", data["image_frames"].shape)
print("single decoded frame shape =", data["image_frames"][0, 0].shape)
print("image_frames.dtype =", data["image_frames"].dtype)

with stage_timer("extract_image_frames", profile_stats):
    image_frames = data["image_frames"]

with stage_timer("flatten_frames", profile_stats):
    flat_frames = image_frames.flatten(0, 1)

with stage_timer("create_message", profile_stats):
    messages = helper.create_message(flat_frames)

print("=== AFTER FLATTEN ===")
print("flat_frames.shape =", flat_frames.shape)
print("flat first frame shape =", flat_frames[0].shape)

model = AlpamayoR1.from_pretrained(
    "nvidia/Alpamayo-R1-10B",
    dtype=torch.bfloat16,
).to("cuda")

processor = helper.get_processor(model.tokenizer)

with stage_timer("apply_chat_template", profile_stats):
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )

print("=== AFTER PROCESSOR ===")
for k, v in inputs.items():
    if hasattr(v, "shape"):
        print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}, device={getattr(v, 'device', 'cpu')}")
    else:
        print(f"{k}: type={type(v)}")

print("=== BEFORE MODEL FORWARD ===")
for k, v in inputs.items():
    if hasattr(v, "shape"):
        print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}, device={getattr(v, 'device', 'cpu')}")

# Profile CPU -> GPU transfer separately for the tokenized VLM inputs
with stage_timer("move_to_cuda", profile_stats):
    tokenized_data = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            tokenized_data[k] = v.to("cuda", non_blocking=False)
        else:
            tokenized_data[k] = v
    torch.cuda.synchronize()

# Keep the rest of the model inputs consistent with the original script
model_inputs = {
    "tokenized_data": tokenized_data,
    "ego_history_xyz": data["ego_history_xyz"],
    "ego_history_rot": data["ego_history_rot"],
}

# Move the remaining non-tokenized tensors using the helper, but avoid re-copying tokenized_data
with stage_timer("move_other_inputs_to_cuda", profile_stats):
    model_inputs["ego_history_xyz"] = helper.to_device(model_inputs["ego_history_xyz"], "cuda")
    model_inputs["ego_history_rot"] = helper.to_device(model_inputs["ego_history_rot"], "cuda")
    torch.cuda.synchronize()

torch.cuda.manual_seed_all(42)

with stage_timer("gpu_rollout", profile_stats):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
        )
    torch.cuda.synchronize()

total_profiled_ms = sum(row["wall_ms"] for row in profile_stats)
print(f"\nTotal profiled wall time: {total_profiled_ms:.2f} ms")

# the size is [batch_size, num_traj_sets, num_traj_samples]
print("Chain-of-Causation (per trajectory):\n", extra["cot"][0])

gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
min_ade = diff.min()
print("minADE:", min_ade, "meters")
print(
    "Note: VLA-reasoning models produce nondeterministic outputs due to trajectory sampling, "
    "hardware differences, etc. With num_traj_samples=1 (set for GPU memory compatibility), "
    "variance in minADE is expected. For visual sanity checks, see notebooks/inference.ipynb"
)

print("\n=== PROFILE SUMMARY ===")
print(json.dumps(profile_stats, indent=2))
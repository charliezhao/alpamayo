#!/usr/bin/env python3
import os
import sys
import time
import json
import argparse
import socket
import numpy as np
import torch
from tqdm import tqdm
import traceback

class NvtxRange:
    def __init__(self, name): self.name = name
    def __enter__(self): torch.cuda.nvtx.range_push(self.name)
    def __exit__(self, *args): torch.cuda.nvtx.range_pop()

# Ensure src/ layout works
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper
import physical_ai_av


# -------------------------
# Utilities
# -------------------------
def json_safe(x):
    """Convert objects to JSON-serializable types."""
    import numpy as _np
    import torch as _torch

    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, _np.ndarray):
        return x.tolist()
    if isinstance(x, _torch.Tensor):
        return x.detach().cpu().tolist()
    return str(x)


def compute_minade(data, pred_xyz):
    """
    Same minADE logic as test_inference.py
    """
    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = (
        pred_xyz.detach()
        .cpu()
        .numpy()[0, 0, :, :, :2]
        .transpose(0, 2, 1)
    )
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    return float(diff.min())

#added on 02/21/2026 by Charlie & Gemini
"""
def inject_nvtx_hooks(model: torch.nn.Module, target_module_name: str, nvtx_label: str):
    # Dynamically wraps a submodule with NVTX markers.
    for name, module in model.named_modules():
        if name == target_module_name:
            def pre_hook(mod, inputs):
                torch.cuda.nvtx.range_push(nvtx_label)
            def post_hook(mod, inputs, outputs):
                torch.cuda.nvtx.range_pop()
            
            module.register_forward_pre_hook(pre_hook)
            module.register_forward_hook(post_hook)
            print(f"[Profiling] Successfully injected '{nvtx_label}' hooks into '{name}'")
            return
    print(f"[Profiling] Warning: Could not find module '{target_module_name}'")
"""
def inject_nvtx_hooks(model: torch.nn.Module, target_module_name: str, nvtx_label: str):
    # dynamically wraps a submodule and manipulates the NVTX stack for a flat timeline
    for name, module in model.named_modules():
        if name == target_module_name:
            
            def pre_hook(mod, inputs):
                # 1. Pop the parent '2_Prefilling' range off the stack to pause it
                torch.cuda.nvtx.range_pop() 
                
                # 2. Push our isolated Vision Encoding range
                torch.cuda.nvtx.range_push(nvtx_label)
                
            def post_hook(mod, inputs, outputs):
                # 3. Pop our Vision Encoding range to end it
                torch.cuda.nvtx.range_pop()
                
                # 4. Push a new range for the remainder of the prefill phase
                torch.cuda.nvtx.range_push("2_Prefilling")
                
            module.register_forward_pre_hook(pre_hook)
            module.register_forward_hook(post_hook)
            
            print(f"[Profiling] Successfully injected '{nvtx_label}' stack-swap hooks into '{name}'")
            return
            
    print(f"[Profiling] Warning: Could not find module '{target_module_name}'")

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_ids_file", required=True, help="newline-separated clip_ids")
    ap.add_argument("--max_clips", type=int, default=100)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--t0_us", type=int, default=5_100_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--top_p", type=float, default=0.98)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--num_traj_samples", type=int, default=1)
    ap.add_argument("--max_generation_length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--maybe_stream", action="store_true")
    ap.add_argument(
        "--model_path",
        type=str,
        default="nvidia/Alpamayo-R1-10B",
        help="HF repo id or local path for Alpamayo model"
    ) 

    args = ap.parse_args()

    # -------------------------
    # Environment metadata (constant per run)
    # -------------------------
    host = socket.gethostname()
    torch_version = torch.__version__
    cuda_version = torch.version.cuda
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    # Precision
    amp_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    model_dtype = amp_dtype

    # Load clip list
    with open(args.clip_ids_file) as f:
        clip_ids = [ln.strip() for ln in f if ln.strip()]
    clip_ids = clip_ids[: args.max_clips]

    # --- ADD THIS: Resume logic to skip processed clips ---
    processed_clips = set()
    if os.path.exists(args.out_jsonl):
        with open(args.out_jsonl, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_clips.add(data['clip_id'])
                except json.JSONDecodeError:
                    continue
    print(f"Skipping {len(processed_clips)} already processed clips.")
    # ------------------------------------------------------    

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)

    # Dataset interface (reuse for speed)
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    """    
    # original code
    # Load model ONCE
    model = AlpamayoR1.from_pretrained(
        args.model_path, 
        dtype=model_dtype
    ).to(args.device)
    #end original code
    """
    # start charlie's hack
    # Load model ONCE
    model = AlpamayoR1.from_pretrained(
        args.model_path, 
        dtype=model_dtype
        ,ignore_mismatched_sizes=True,
    )
    
    # This allocates actual memory (random numbers) for the "meta" tensors
    model.to_empty(device=args.device) #charlie

    # 3. Now move the rest of the initialized weights to the device
    model.to(args.device) 
    #end charlie's hack
    
    processor = helper.get_processor(model.tokenizer)

    # added on 02/21/2026 by Charlie & Gemini
    # --- ADD THIS: Inject the Vision Encoder NVTX Hooks ---
    inject_nvtx_hooks(
        model, 
        target_module_name="vlm.model.visual", 
        nvtx_label="1_3_Vision_Encoding"
    )
    # ------------------------------------------------------

    print("MODEL DTYPE CHECK")
    print("param dtype:", next(model.parameters()).dtype)
    print("bf16 supported:", torch.cuda.is_bf16_supported())
    print("flash sdp enabled:", torch.backends.cuda.flash_sdp_enabled())
    print("mem_efficient sdp enabled:", torch.backends.cuda.mem_efficient_sdp_enabled())
    print("math sdp enabled:", torch.backends.cuda.math_sdp_enabled())

    torch.cuda.manual_seed_all(args.seed)

    # -------------------------
    # Warmup (not recorded)
    # -------------------------
    warm_id = clip_ids[0]
    data = load_physical_aiavdataset(
        warm_id, t0_us=args.t0_us, avdi=avdi, maybe_stream=args.maybe_stream
    )
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, args.device)

    with torch.autocast(args.device, dtype=amp_dtype):
        _ = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=args.top_p,
            temperature=args.temperature,
            num_traj_samples=args.num_traj_samples,
            max_generation_length=args.max_generation_length,
            return_extra=False,
        )
    torch.cuda.synchronize()

    #added on 02/21/2026 by Charlie & Gemini
    # --- ADD THIS: Start profiler only for the actual main loop ---
    torch.cuda.cudart().cudaProfilerStart()
    # --------------------------------------------------------------

    # -------------------------
    # Main loop (JSONL output)
    # -------------------------
    mode = "a" if os.path.exists(args.out_jsonl) else "w"
    with open(args.out_jsonl, mode) as out:
        for cid in tqdm(clip_ids, desc="Clips"):

            # --- ADD THIS: Skip if already done ---
            if cid in processed_clips:
                continue
            # --------------------------------------
            rec = {
                # identification
                "clip_id": cid,
                "t0_us": args.t0_us,

                # environment
                "host": host,
                "torch_version": torch_version,
                "cuda_version": cuda_version,
                "gpu_name": gpu_name,

                # config
                "dtype": args.dtype,
                "num_traj_samples": args.num_traj_samples,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "max_generation_length": args.max_generation_length,
            }

            t_clip0 = time.time()
            try:
                data = load_physical_aiavdataset(
                    cid,
                    t0_us=args.t0_us,
                    avdi=avdi,
                    maybe_stream=args.maybe_stream,
                )

                messages = helper.create_message(data["image_frames"].flatten(0, 1))
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    continue_final_message=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                model_inputs = {
                    "tokenized_data": inputs,
                    "ego_history_xyz": data["ego_history_xyz"],
                    "ego_history_rot": data["ego_history_rot"],
                }
                model_inputs = helper.to_device(model_inputs, args.device)

                torch.cuda.synchronize()
                start = torch.cuda.Event(True); end = torch.cuda.Event(True)
                torch.cuda.reset_peak_memory_stats()
                start.record()
                t0 = time.time()
                """
                with NvtxRange(f"alpamayo_clip:{cid}"):
                    with torch.autocast(args.device, dtype=amp_dtype):
                        pred_xyz, pred_rot, extra = (
                            model.sample_trajectories_from_data_with_vlm_rollout(
                                data=model_inputs,
                                top_p=args.top_p,
                                temperature=args.temperature,
                                num_traj_samples=args.num_traj_samples,
                                max_generation_length=args.max_generation_length,
                                return_extra=True,
                            )
                        )
                """
                with NvtxRange(f"alpamayo_clip:{cid}"):
                    with torch.autocast(args.device, dtype=amp_dtype):
                        ret = model.sample_trajectories_from_data_with_vlm_rollout(
                            data=model_inputs,
                            top_p=args.top_p,
                            temperature=args.temperature,
                            num_traj_samples=args.num_traj_samples,
                            max_generation_length=args.max_generation_length,
                            return_extra=True,
                        )

                        if isinstance(ret, tuple) and len(ret) == 3:
                            pred_xyz, pred_rot, extra = ret
                        elif isinstance(ret, tuple) and len(ret) == 2:
                            pred_xyz, pred_rot = ret
                            extra = {}  # or None
                        else:
                            raise TypeError(
                                f"Unexpected return from sample_trajectories_from_data_with_vlm_rollout: "
                                f"type={type(ret)} value={ret!r}"
                            )                

                end.record()
                torch.cuda.synchronize()
                t1 = time.time()

                rec["ok"] = True
                rec["gpu_rollout_s"] = t1 - t0
                rec["gpu_rollout_kernel_ms"] = start.elapsed_time(end)
                rec["e2e_clip_s"] = time.time() - t_clip0
                rec["minADE_m"] = compute_minade(data, pred_xyz)
                rec["max_alloc_gb"] = torch.cuda.max_memory_allocated() / 1e9
                rec["max_rsv_gb"] = torch.cuda.max_memory_reserved()/1e9

                cot = extra.get("cot", None)
                if cot is not None:
                    try:
                        rec["cot"] = cot[0]
                    except Exception:
                        rec["cot"] = cot
                else:
                    rec["cot"] = None

            except Exception as e:
                rec["ok"] = False
                rec["e2e_clip_s"] = time.time() - t_clip0
                # rec["error"] = repr(e)
                rec["error"] = traceback.format_exc()

            out.write(json.dumps(json_safe(rec)) + "\n")
            out.flush()
            
            # --- ADD THIS: Clear fragmentation at the end of every clip ---
            torch.cuda.empty_cache() 
            # -------------------------------------------------------------

    print(f"Wrote JSONL: {args.out_jsonl}")
    
    #added on 02/21/2026 by Charlie & Gemini
    # --- ADD THIS: Stop the profiler ---
    torch.cuda.cudart().cudaProfilerStop()
    # -----------------------------------

if __name__ == "__main__":
    main()


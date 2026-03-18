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

from torch.profiler import profile, ProfilerActivity

class NvtxRange:
    def __init__(self, name): self.name = name
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(self.name)
    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()

# Ensure src/ layout works
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper
import physical_ai_av


def json_safe(x):
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
    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = (
        pred_xyz.detach()
        .cpu()
        .numpy()[0, 0, :, :, :2]
        .transpose(0, 2, 1)
    )
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    return float(diff.min())


def inject_nvtx_hooks(model: torch.nn.Module, target_module_name: str, nvtx_label: str):
    # dynamically wraps a submodule and manipulates the NVTX stack for a flat timeline
    for name, module in model.named_modules():
        if name == target_module_name:

            def pre_hook(mod, inputs):
                # Pop parent range to "pause" it (your original behavior)
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_push(nvtx_label)

            def post_hook(mod, inputs, outputs):
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_push("2_Prefilling")

            module.register_forward_pre_hook(pre_hook)
            module.register_forward_hook(post_hook)
            print(f"[Profiling] Injected '{nvtx_label}' stack-swap hooks into '{name}'")
            return

    print(f"[Profiling] Warning: Could not find module '{target_module_name}'")


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _key_averages_to_rows(prof):
    """
    Convert torch.profiler key_averages() to JSON-friendly rows.
    Times are in microseconds (as returned by torch.profiler).
    """
    rows = []
    for it in prof.key_averages():
        rows.append({
            "key": it.key,                       # e.g. "aten::mm"
            "count": int(it.count),
            "self_cpu_time_total_us": float(it.self_cpu_time_total),
            "cpu_time_total_us": float(it.cpu_time_total),
            "self_cuda_time_total_us": float(getattr(it, "self_cuda_time_total", 0.0)),
            "cuda_time_total_us": float(getattr(it, "cuda_time_total", 0.0)),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_ids_file", required=True, help="newline-separated clip_ids")
    ap.add_argument("--max_clips", type=int, default=100)
    ap.add_argument("--t0_us", type=int, default=5_100_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--top_p", type=float, default=0.98)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--num_traj_samples", type=int, default=1)
    ap.add_argument("--max_generation_length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--maybe_stream", action="store_true")
    ap.add_argument("--model_path", type=str, default="nvidia/Alpamayo-R1-10B")

    # New outputs for op stats + traces
    ap.add_argument("--opstats_out_jsonl", required=True,
                    help="JSONL output: one line per profiled clip with aten::* op counts+times")
    ap.add_argument("--trace_dir", default="torch_traces",
                    help="Directory to write per-clip Chrome/Perfetto traces (trace_<clipid>.json)")
    ap.add_argument("--profile_clips", type=int, default=1,
                    help="How many clips to torch.profile (default 1). Others will run without op profiling.")
    ap.add_argument("--row_limit_print", type=int, default=50,
                    help="Print top ops for each profiled clip (0 to disable printing).")
    ap.add_argument("--sort_by", default="self_cuda_time_total",
                    help="Sort key for printed table (e.g., self_cuda_time_total, cuda_time_total, cpu_time_total).")

    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.opstats_out_jsonl) or ".", exist_ok=True)
    os.makedirs(args.trace_dir, exist_ok=True)

    host = socket.gethostname()
    torch_version = torch.__version__
    cuda_version = torch.version.cuda
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    amp_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    model_dtype = amp_dtype

    with open(args.clip_ids_file) as f:
        clip_ids = [ln.strip() for ln in f if ln.strip()]
    clip_ids = clip_ids[: args.max_clips]

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    model = AlpamayoR1.from_pretrained(
        args.model_path,
        dtype=model_dtype
    ).to(args.device)

    processor = helper.get_processor(model.tokenizer)

    # Keep your vision-encoder NVTX stack-swap hook
    inject_nvtx_hooks(
        model,
        target_module_name="vlm.model.visual",
        nvtx_label="1_3_Vision_Encoding"
    )

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
    _sync()

    # -------------------------
    # Main loop
    # -------------------------
    activities = [ProfilerActivity.CPU, ProfilerActivity.GPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with open(args.opstats_out_jsonl, "w") as out:
        profiled = 0
        for cid in tqdm(clip_ids, desc="Clips"):
            # Load inputs (same as your script)
            t_clip0 = time.time()
            rec_meta = {
                "clip_id": cid,
                "t0_us": args.t0_us,
                "host": host,
                "torch_version": torch_version,
                "cuda_version": cuda_version,
                "gpu_name": gpu_name,
                "dtype": args.dtype,
                "num_traj_samples": args.num_traj_samples,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "max_generation_length": args.max_generation_length,
            }

            try:
                data = load_physical_aiavdataset(
                    cid, t0_us=args.t0_us, avdi=avdi, maybe_stream=args.maybe_stream
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

                # Decide whether to profile this clip
                do_profile = (profiled < args.profile_clips)

                if do_profile:
                    with profile(
                        activities=activities,
                        record_shapes=False,
                        profile_memory=False,
                        with_stack=False,
                        with_modules=False,
                    ) as prof:
                        _sync()
                        with torch.profiler.record_function(f"alpamayo_clip:{cid}"):
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
                        _sync()

                    # Save trace for UI filtering by NVTX ranges / record_function
                    trace_path = os.path.join(args.trace_dir, f"trace_{cid}.json")
                    prof.export_chrome_trace(trace_path)

                    # Op stats (aten::*)
                    rows = _key_averages_to_rows(prof)

                    # Build output record
                    pred_xyz = ret[0] if isinstance(ret, tuple) and len(ret) >= 1 else None
                    extra = ret[2] if isinstance(ret, tuple) and len(ret) == 3 else {}
                    cot = extra.get("cot", None) if isinstance(extra, dict) else None

                    rec = dict(rec_meta)
                    rec["ok"] = True
                    rec["e2e_clip_s"] = time.time() - t_clip0
                    rec["minADE_m"] = compute_minade(data, pred_xyz) if pred_xyz is not None else None
                    rec["trace_path"] = trace_path
                    rec["opstats"] = rows
                    rec["cot"] = (cot[0] if isinstance(cot, (list, tuple)) and len(cot) > 0 else cot)

                    out.write(json.dumps(json_safe(rec)) + "\n")
                    out.flush()

                    # Optional: print a quick top table to console
                    if args.row_limit_print and args.row_limit_print > 0:
                        print(f"\n=== {cid} top ops (whole clip) ===")
                        print(prof.key_averages().table(
                            sort_by=args.sort_by,
                            row_limit=args.row_limit_print
                        ))

                    profiled += 1
                else:
                    # Not profiled: just run (no opstats output line) to keep script simple
                    with NvtxRange(f"alpamayo_clip:{cid}"):
                        with torch.autocast(args.device, dtype=amp_dtype):
                            _ = model.sample_trajectories_from_data_with_vlm_rollout(
                                data=model_inputs,
                                top_p=args.top_p,
                                temperature=args.temperature,
                                num_traj_samples=args.num_traj_samples,
                                max_generation_length=args.max_generation_length,
                                return_extra=False,
                            )
                    _sync()

            except Exception:
                # For profiled clips, still write an error line so you know which failed
                rec = dict(rec_meta)
                rec["ok"] = False
                rec["e2e_clip_s"] = time.time() - t_clip0
                rec["error"] = traceback.format_exc()
                out.write(json.dumps(json_safe(rec)) + "\n")
                out.flush()

            # reduce fragmentation between clips
            torch.cuda.empty_cache()

    print(f"Wrote opstats JSONL: {args.opstats_out_jsonl}")
    print(f"Wrote traces under: {args.trace_dir}")
    print("Open trace_*.json in https://ui.perfetto.dev and search for 'alpamayo_clip:' or your NVTX labels (e.g. '1_3_Vision_Encoding').")


if __name__ == "__main__":
    main()
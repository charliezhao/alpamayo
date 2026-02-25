#!/usr/bin/env python3
import os, sys, time, json, argparse, socket, traceback
import numpy as np
import torch
from tqdm import tqdm

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
    pred_xy = pred_xyz.detach().cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    return float(diff.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_ids_file", required=True)
    ap.add_argument("--max_clips", type=int, default=100)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--t0_us", type=int, default=5_100_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--top_p", type=float, default=0.98)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--num_traj_samples", type=int, default=1)
    ap.add_argument("--max_generation_length", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--maybe_stream", action="store_true")
    ap.add_argument("--model_path", type=str, default="nvidia/Alpamayo-R1-10B")
    args = ap.parse_args()

    host = socket.gethostname()
    torch_version = torch.__version__
    cuda_version = torch.version.cuda
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    with open(args.clip_ids_file) as f:
        clip_ids = [ln.strip() for ln in f if ln.strip()][: args.max_clips]

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    # ✅ ORIGINAL, SAFE load
    model = AlpamayoR1.from_pretrained(args.model_path, dtype=amp_dtype).to(args.device)
    processor = helper.get_processor(model.tokenizer)

    torch.cuda.manual_seed_all(args.seed)

    # ✅ Warmup matches main call (including return_extra)
    warm_id = clip_ids[0]
    data = load_physical_aiavdataset(warm_id, t0_us=args.t0_us, avdi=avdi, maybe_stream=args.maybe_stream)
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = helper.to_device(
        {"tokenized_data": inputs, "ego_history_xyz": data["ego_history_xyz"], "ego_history_rot": data["ego_history_rot"]},
        args.device,
    )

    with torch.autocast(args.device, dtype=amp_dtype):
        _ = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=args.top_p,
            temperature=args.temperature,
            num_traj_samples=args.num_traj_samples,
            max_generation_length=args.max_generation_length,
            return_extra=True,   # match
        )
    torch.cuda.synchronize()

    mode = "a" if os.path.exists(args.out_jsonl) else "w"
    with open(args.out_jsonl, mode) as out:
        for cid in tqdm(clip_ids, desc="Clips"):
            rec = {
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

            t_clip0 = time.time()
            try:
                data = load_physical_aiavdataset(cid, t0_us=args.t0_us, avdi=avdi, maybe_stream=args.maybe_stream)
                messages = helper.create_message(data["image_frames"].flatten(0, 1))
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    continue_final_message=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                model_inputs = helper.to_device(
                    {"tokenized_data": inputs, "ego_history_xyz": data["ego_history_xyz"], "ego_history_rot": data["ego_history_rot"]},
                    args.device,
                )

                torch.cuda.synchronize()
                start = torch.cuda.Event(True); end = torch.cuda.Event(True)
                torch.cuda.reset_peak_memory_stats()

                start.record()
                t0 = time.time()
                with torch.autocast(args.device, dtype=amp_dtype):
                    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                        data=model_inputs,
                        top_p=args.top_p,
                        temperature=args.temperature,
                        num_traj_samples=args.num_traj_samples,
                        max_generation_length=args.max_generation_length,
                        return_extra=True,
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
                rec["max_rsv_gb"] = torch.cuda.max_memory_reserved() / 1e9

                cot = extra.get("cot", None) if isinstance(extra, dict) else None
                rec["cot"] = cot[0] if (cot is not None and isinstance(cot, (list, tuple)) and len(cot) > 0) else cot

            except Exception:
                rec["ok"] = False
                rec["e2e_clip_s"] = time.time() - t_clip0
                rec["error"] = traceback.format_exc()

            out.write(json.dumps(json_safe(rec)) + "\n")
            out.flush()

    print(f"Wrote JSONL: {args.out_jsonl}")


if __name__ == "__main__":
    main()
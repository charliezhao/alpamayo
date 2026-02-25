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
from torch.profiler import record_function, profile, ProfilerActivity
from collections import defaultdict

# added to profile layer infor. 2026/02/23
def _tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()

def _collect_tensors(obj, out_list):
    if torch.is_tensor(obj):
        out_list.append(obj)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            _collect_tensors(x, out_list)
    elif isinstance(obj, dict):
        for x in obj.values():
            _collect_tensors(x, out_list)

def _summarize_io(obj):
    ts = []
    _collect_tensors(obj, ts)
    desc = []
    total = 0
    for t in ts:
        b = _tensor_bytes(t)
        desc.append({
            "shape": list(t.shape),
            "dtype": str(t.dtype).replace("torch.", ""),
            "device": str(t.device),
            "bytes": int(b),
        })
        total += b
    return desc, int(total)

def _prod(xs):
    p = 1
    for x in xs:
        p *= int(x)
    return p

def _first_desc(desc_list):
    return desc_list[0] if desc_list else None

def infer_mac_fields(module_type: str, module_name: str, in_desc, out_desc):
    """
    Returns: (mac_like: bool, mac_kind: str, est_macs: float|None)

    mac_kind in:
      "linear" | "conv" | "attn" | "embedding_gather" | "elementwise" | "unknown"
    """
    t = (module_type or "").lower()
    m = (module_name or "").lower()
    in0 = _first_desc(in_desc)
    out0 = _first_desc(out_desc)

    if t == "linear":
        # MACs ~ batch * K * N
        if in0 and out0 and in0.get("shape") and out0.get("shape"):
            ish = in0["shape"]
            osh = out0["shape"]
            if len(ish) >= 1 and len(osh) >= 1:
                K = int(ish[-1])
                N = int(osh[-1])
                batch = _prod(ish[:-1]) if len(ish) > 1 else 1
                return True, "linear", float(batch * K * N)
        return True, "linear", None

    if "conv" in t:
        # MAC-heavy but kernel params not present in JSONL => est_macs unknown
        return True, "conv", None

    if t == "embedding":
        return False, "embedding_gather", None

    if "attention" in t or ".attn" in m:
        # composite attention often hides fused kernels
        return True, "attn", None

    if any(k in t for k in ["gelu", "silu", "relu", "tanh", "sigmoid"]):
        return False, "elementwise", None

    return False, "unknown", None

def default_filter(name: str, mod: torch.nn.Module) -> bool:
    n = name.lower()
    t = type(mod).__name__.lower()

    # skip tiny/noisy modules
    if t in {"dropout", "identity", "relu", "gelu", "silu"}:
        return False

    keep_keys = [
        "vlm", "vision", "encoder", "decoder",
        "transformer", "block", "attn", "attention",
        "mlp", "ffn", "proj", "projector",
        "head", "action",
    ]
    return any(k in n for k in keep_keys)

class ExclusiveModuleProfiler:
    """
    Per-call inclusive/exclusive timing with a call stack.

    Adds torch.profiler ranges:
        MOD::<module_path>
    so you can inspect operator-level details inside composites like Attention/MLP/Block.
    """
    def __init__(self, model, name_filter=None, max_records=200000, cuda_sync=True):
        self.model = model
        self.name_filter = name_filter
        self.max_records = max_records
        self.cuda_sync = cuda_sync
        self.records = []
        self._handles = []
        self.use_cuda = torch.cuda.is_available()
        self._stack = []  # frames
        self._global_call_id = 0
        self._per_module_call_id = defaultdict(int)

    def _should_keep(self, name, mod):
        if name == "":
            return False
        if self.name_filter is None:
            return True
        return self.name_filter(name, mod)

    def start(self):
        for name, mod in self.model.named_modules():
            if not self._should_keep(name, mod):
                continue

            def pre_hook(_mod, inputs, _name=name):
                if len(self.records) >= self.max_records:
                    return

                # assign IDs
                self._global_call_id += 1
                gid = self._global_call_id

                mid = self._per_module_call_id[_name]
                self._per_module_call_id[_name] += 1

                frame = {
                    "module": _name,
                    "type": type(_mod).__name__,
                    "global_call_id": gid,     # unique across whole run
                    "module_call_idx": mid,    # 0,1,2... per module path
                    "t0": time.perf_counter(),
                    "child_gpu_ms_sum": 0.0,
                    "child_cpu_ms_sum": 0.0,
                }

                # Begin a module range for torch.profiler trace (unique!)
                rf = record_function(f"MOD::{_name}#gid={gid}#mid={mid}")
                rf.__enter__()
                frame["rf"] = rf

                if self.use_cuda:
                    frame["e0"] = torch.cuda.Event(enable_timing=True)
                    frame["e1"] = torch.cuda.Event(enable_timing=True)
                    frame["e0"].record()

                in_desc, in_bytes = _summarize_io(inputs)
                frame["in_desc"] = in_desc
                frame["in_bytes"] = int(in_bytes)

                self._stack.append(frame)

            def post_hook(_mod, inputs, outputs, _name=name):
                if not self._stack:
                    return
                # Must pop to keep stack consistent even if record limit hit
                frame = self._stack.pop()

                t1 = time.perf_counter()
                out_desc, out_bytes = _summarize_io(outputs)

                gpu_inclusive = None
                if self.use_cuda:
                    frame["e1"].record()

                    if self.cuda_sync:
                        # Old behavior: stable but slow
                        torch.cuda.synchronize()
                    else:
                        # New behavior: ensure THIS end event is complete (much cheaper than full sync)
                        frame["e1"].synchronize()

                    gpu_inclusive = frame["e0"].elapsed_time(frame["e1"])

                cpu_inclusive = (t1 - frame["t0"]) * 1000.0

                # Exclusive = Inclusive - sum(children inclusive)
                gpu_exclusive = None
                cpu_exclusive = cpu_inclusive - frame["child_cpu_ms_sum"]
                if gpu_inclusive is not None:
                    gpu_exclusive = gpu_inclusive - frame["child_gpu_ms_sum"]
                    if gpu_exclusive < 0:
                        gpu_exclusive = 0.0

                # End module range
                rf = frame.get("rf", None)
                if rf is not None:
                    rf.__exit__(None, None, None)

                rec = {
                    "module": frame["module"],
                    "type": frame["type"],
                    "in_bytes": frame["in_bytes"],
                    "out_bytes": int(out_bytes),
                    "cpu_ms_inclusive": float(cpu_inclusive),
                    "cpu_ms_exclusive": float(cpu_exclusive),
                    "gpu_ms_inclusive": None if gpu_inclusive is None else float(gpu_inclusive),
                    "gpu_ms_exclusive": None if gpu_exclusive is None else float(gpu_exclusive),
                    "in_desc": frame.get("in_desc", []),
                    "out_desc": out_desc,
                }

                # Optional: add MAC fields (see section C below)
                mac_like, mac_kind, est_macs = infer_mac_fields(
                    module_type=rec["type"],
                    module_name=rec["module"],
                    in_desc=rec["in_desc"],
                    out_desc=rec["out_desc"],
                )
                rec["mac_like"] = mac_like
                rec["mac_kind"] = mac_kind
                rec["est_macs"] = est_macs
                rec["global_call_id"] = frame["global_call_id"]
                rec["module_call_idx"] = frame["module_call_idx"]

                self.records.append(rec)

                # Add this frame’s inclusive time to parent’s child sum
                if self._stack:
                    parent = self._stack[-1]
                    parent["child_cpu_ms_sum"] += cpu_inclusive
                    if gpu_inclusive is not None:
                        parent["child_gpu_ms_sum"] += gpu_inclusive

            self._handles.append(mod.register_forward_pre_hook(pre_hook))
            self._handles.append(mod.register_forward_hook(post_hook))

    def stop(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        self._stack = []

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
    # added 2026/02/23
    ap.add_argument("--layer_profile_out", type=str, default=None,
                    help="If set, write per-module timing+IO info to this JSONL.")
    ap.add_argument("--layer_profile_cuda_sync", action="store_true",
                    help="Sync after each module for stable gpu_ms (slower).")
    ap.add_argument("--torch_profile", action="store_true",
                help="Enable torch.profiler and export a chrome trace (1 clip recommended).")
    ap.add_argument("--torch_profile_out", type=str, default="../alpamayo_outputs/torch_trace.json",
                help="Output chrome trace path.")

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


                # added 2026/02/23
                prof = None
                if args.layer_profile_out:
                    prof = ExclusiveModuleProfiler(
                        model,
                        name_filter=lambda n, m: default_filter(n, m),
                        cuda_sync=args.layer_profile_cuda_sync,
                        max_records=200000,
                    )
                    prof._global_call_id = 0
                    prof._per_module_call_id.clear()
                    prof.start()

                # -------------------------
                # Forward (optionally torch.profiler)
                # -------------------------
                def _run_forward():
                    with torch.autocast(args.device, dtype=amp_dtype):
                        return model.sample_trajectories_from_data_with_vlm_rollout(
                            data=model_inputs,
                            top_p=args.top_p,
                            temperature=args.temperature,
                            num_traj_samples=args.num_traj_samples,
                            max_generation_length=args.max_generation_length,
                            return_extra=True,
                        )

                if args.torch_profile:
                    # NOTE: do 1 clip when torch_profile is enabled; it adds overhead.
                    with profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        record_shapes=True,
                        profile_memory=True,
                    ) as prof_torch:
                        ret = _run_forward()

                    os.makedirs(os.path.dirname(args.torch_profile_out) or ".", exist_ok=True)
                    prof_torch.export_chrome_trace(args.torch_profile_out)
                    print(f"[torch.profiler] wrote chrome trace: {args.torch_profile_out}")
                else:
                    ret = _run_forward()

                # Unpack return values safely
                if isinstance(ret, tuple) and len(ret) == 3:
                    pred_xyz, pred_rot, extra = ret
                elif isinstance(ret, tuple) and len(ret) == 2:
                    pred_xyz, pred_rot = ret
                    extra = {}
                else:
                    raise TypeError(
                        f"Unexpected return from sample_trajectories_from_data_with_vlm_rollout: "
                        f"type={type(ret)} value={ret!r}"
                    )

                # added 2026/02/23
                if prof is not None:
                    prof.stop()
                    # write JSONL (append per clip)
                    with open(args.layer_profile_out, "a") as f:
                        for r in prof.records:
                            r["clip_id"] = cid
                            f.write(json.dumps(r) + "\n")

                    # quick top-20 print for this clip
                    recs = [r for r in prof.records if r.get("gpu_ms_exclusive") is not None]
                    recs.sort(key=lambda x: x["gpu_ms_exclusive"], reverse=True)
                    print("\n[LayerProfile] Top 20 modules by gpu_ms_exclusive:")
                    for r in recs[:20]:
                        print(
                            f'{r["gpu_ms_exclusive"]:9.3f} ms (incl {r.get("gpu_ms_inclusive", 0.0):9.3f} ms)  '
                            f'{r["in_bytes"]/1e6:8.1f}MB -> {r["out_bytes"]/1e6:8.1f}MB  '
                            f'{r["type"]:24s}  {r["module"]}'
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

if __name__ == "__main__":
    main()


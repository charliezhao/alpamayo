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
import re
from collections import defaultdict

# added on 02/26/2026
import re
from collections import defaultdict

def _load_chrome_trace_events(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "traceEvents" in data:
        return data["traceEvents"]
    if isinstance(data, list):
        return data
    raise ValueError("Unknown chrome trace JSON format")

_MOD_RE = re.compile(r"^MOD::(?P<mod>.+?)#gid=(?P<gid>\d+)#mid=(?P<mid>\d+)$")

def _is_mod_slice(ev):
    return ev.get("ph") == "X" and isinstance(ev.get("name"), str) and ev["name"].startswith("MOD::")

def _parse_mod_name(name: str):
    # MOD::path#gid=123#mid=0
    m = _MOD_RE.match(name)
    if not m:
        return None
    return m.group("mod"), int(m.group("gid")), int(m.group("mid"))

def _is_cuda_kernel_event(ev):
    """
    Heuristic: CUDA kernels are 'X' events that are not MOD slices and look like kernel names.
    Torch traces vary; we combine cat + name heuristics.
    """
    if ev.get("ph") != "X":
        return False
    name = ev.get("name")
    if not isinstance(name, str) or name.startswith("MOD::"):
        return False

    cat = ev.get("cat", "")
    if isinstance(cat, str):
        cl = cat.lower()
        if "kernel" in cl or "cuda" in cl or "gpu" in cl:
            return True

    # name hints seen in practice
    hints = ("nvjet_", "cublas", "cudnn", "cutlass", "triton", "flash", "vectorized_elementwise", "void ")
    return name.startswith(hints) or any(h in name for h in hints)

def _interval_overlap_us(a0, a1, b0, b1):
    lo = max(a0, b0)
    hi = min(a1, b1)
    return hi - lo if hi > lo else 0.0

def enrich_jsonl_with_trace_gpu(
    trace_path: str,
    layers_jsonl_in: str,
    layers_jsonl_out: str,
    compute_exclusive: bool = True,
):
    """
    Enrich module JSONL with GPU timings from torch chrome trace.

    Correct attribution:
      - Use GPU MOD slices (cat == 'gpu_user_annotation') as the module interval.
      - Attribute a kernel to a module if kernel.start_ts is inside the module interval.
        (NOT overlap, NOT CPU interval)

    Exclusive:
      - Build a true nesting tree using interval containment on GPU MOD slices.
      - exclusive = inclusive - sum(direct_children_inclusive)
    """
    events = _load_chrome_trace_events(trace_path)

    # -------------------------
    # 1) Collect GPU MOD slices keyed by (gid, mid)
    # -------------------------
    # In torch traces, MOD slices appear on CPU (user_annotation) and GPU (gpu_user_annotation).
    # We want the GPU ones.
    mod_gpu = {}  # (gid,mid) -> {"module":path, "ts":ts, "te":te, "pid":pid, "tid":tid}
    for ev in events:
        if not _is_mod_slice(ev):
            continue
        if ev.get("cat") != "gpu_user_annotation":
            continue

        parsed = _parse_mod_name(ev.get("name", ""))
        if not parsed:
            continue
        mod_path, gid, mid = parsed

        ts = float(ev.get("ts", 0.0))
        dur = float(ev.get("dur", 0.0))
        if dur <= 0:
            continue
        te = ts + dur

        key = (gid, mid)
        prev = mod_gpu.get(key)

        # If duplicates exist, keep the longer GPU interval (usually the "real" one)
        if prev is None or (te - ts) > (prev["te"] - prev["ts"]):
            mod_gpu[key] = {
                "module": mod_path,
                "ts": ts,
                "te": te,
                "pid": ev.get("pid", None),
                "tid": ev.get("tid", None),
            }

    if not mod_gpu:
        raise RuntimeError(
            f"No GPU MOD::...#gid= #mid= slices found in trace (gpu_user_annotation): {trace_path}"
        )

    # -------------------------
    # 2) Collect CUDA kernel events with pid/tid (stream info)
    # -------------------------
    kernels = []
    for ev in events:
        if not _is_cuda_kernel_event(ev):
            continue
        ts = float(ev.get("ts", 0.0))
        dur = float(ev.get("dur", 0.0))
        if dur <= 0:
            continue
        kernels.append({
            "ts": ts,
            "te": ts + dur,
            "dur_us": dur,
            "pid": ev.get("pid", None),
            "tid": ev.get("tid", None),
            "name": ev.get("name", ""),
            "cat": ev.get("cat", ""),
        })

    kernels.sort(key=lambda x: x["ts"])

    # Small helper: restrict kernels to same GPU track as the MOD slice (same pid/tid)
    def _same_track(k, mod):
        # If pid/tid missing, fall back to not filtering.
        if mod.get("pid", None) is None or mod.get("tid", None) is None:
            return True
        return (k.get("pid", None) == mod["pid"]) and (k.get("tid", None) == mod["tid"])

    # -------------------------
    # 3) Inclusive GPU time: kernel START inside [ts, te)
    # -------------------------
    gpu_incl_ms = defaultdict(float)

    # For speed, we can scan once with an index, but 1-clip traces are OK with simple logic.
    for key, itv in mod_gpu.items():
        ts, te = itv["ts"], itv["te"]
        s_us = 0.0
        for k in kernels:
            if k["ts"] < ts:
                continue
            if k["ts"] >= te:
                break
            if not _same_track(k, itv):
                continue
            # Count full kernel duration if it starts inside module interval.
            s_us += k["dur_us"]
        gpu_incl_ms[key] = s_us / 1000.0  # us -> ms

    # -------------------------
    # 4) Exclusive GPU time: true nesting tree by interval containment
    # -------------------------
    gpu_excl_ms = {}
    if compute_exclusive:
        # Build nesting per pid/tid (i.e., per CUDA stream track)
        groups = defaultdict(list)  # (pid,tid) -> list of (key, ts, te)
        for key, itv in mod_gpu.items():
            groups[(itv.get("pid", None), itv.get("tid", None))].append((key, itv["ts"], itv["te"]))

        children = defaultdict(list)  # parent_key -> [child_key...]

        for _, items in groups.items():
            # Sort by start, then longer first if same start (helps nesting)
            items.sort(key=lambda x: (x[1], -(x[2]-x[1])))
            stack = []
            for key, ts, te in items:
                # Pop until we find a container
                while stack and not (stack[-1][1] <= ts and te <= stack[-1][2]):
                    stack.pop()
                if stack:
                    parent_key = stack[-1][0]
                    children[parent_key].append(key)
                stack.append((key, ts, te))

        for key, itv in mod_gpu.items():
            incl = gpu_incl_ms.get(key, 0.0)
            child_sum = 0.0
            for ck in children.get(key, []):
                child_sum += gpu_incl_ms.get(ck, 0.0)
            excl = incl - child_sum
            if excl < 0:
                excl = 0.0
            gpu_excl_ms[key] = excl

    # -------------------------
    # 5) Write enriched JSONL
    # -------------------------
    with open(layers_jsonl_in, "r") as fin, open(layers_jsonl_out, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            gid = rec.get("global_call_id", None)
            mid = rec.get("module_call_idx", None)

            if isinstance(gid, int) and isinstance(mid, int):
                key = (gid, mid)
                rec["gpu_ms_inclusive_trace"] = float(gpu_incl_ms.get(key, 0.0))
                rec["gpu_ms_exclusive_trace"] = float(gpu_excl_ms.get(key, 0.0)) if compute_exclusive else None
            else:
                rec["gpu_ms_inclusive_trace"] = None
                rec["gpu_ms_exclusive_trace"] = None

            fout.write(json.dumps(rec) + "\n")

    print(f"[trace post] enriched layer JSONL written: {layers_jsonl_out}")

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

def _tensor_desc(t: torch.Tensor):
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "device": str(t.device),
        "bytes": int(t.numel() * t.element_size()),
    }

def _summarize_params(mod: torch.nn.Module):
    """
    Return (param_desc_list, param_bytes_total, weight_desc, bias_desc)
    """
    desc_list = []
    total = 0
    for pn, p in mod.named_parameters(recurse=False):
        if p is None:
            continue
        d = _tensor_desc(p)
        d["name"] = pn
        desc_list.append(d)
        total += d["bytes"]

    w = getattr(mod, "weight", None)
    b = getattr(mod, "bias", None)

    weight_desc = _tensor_desc(w) if torch.is_tensor(w) else None
    bias_desc = _tensor_desc(b) if torch.is_tensor(b) else None

    return desc_list, int(total), weight_desc, bias_desc

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
                    #"child_gpu_ms_sum": 0.0,
                    "child_cpu_ms_sum": 0.0,
                }

                # ---- parameter summary ----
                try:
                    p_desc, p_bytes, w_desc, b_desc = _summarize_params(_mod)
                except Exception:
                    p_desc, p_bytes, w_desc, b_desc = [], 0, None, None

                frame["param_desc"] = p_desc
                frame["param_bytes"] = int(p_bytes)
                frame["weight_desc"] = w_desc
                frame["bias_desc"] = b_desc                

                # Begin a module range for torch.profiler trace (unique!)
                rf = record_function(f"MOD::{_name}#gid={gid}#mid={mid}")
                rf.__enter__()
                frame["rf"] = rf

                # 02/26/2026 ❌ removed CUDA event timing
                #if self.use_cuda:
                #    frame["e0"] = torch.cuda.Event(enable_timing=True)
                #    frame["e1"] = torch.cuda.Event(enable_timing=True)
                #    frame["e0"].record()

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

                # 02/26/2026 remove this block
                """
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
                """

                cpu_inclusive = (t1 - frame["t0"]) * 1000.0

                # Exclusive = Inclusive - sum(children inclusive)
                # gpu_exclusive = None
                cpu_exclusive = cpu_inclusive - frame["child_cpu_ms_sum"]
                if cpu_exclusive < 0:
                    cpu_exclusive = 0.0
                """
                if gpu_inclusive is not None:
                    gpu_exclusive = gpu_inclusive - frame["child_gpu_ms_sum"]
                    if gpu_exclusive < 0:
                        gpu_exclusive = 0.0
                """

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

                    # Changed on 02/26/2026
                    # "gpu_ms_inclusive": None if gpu_inclusive is None else float(gpu_inclusive),
                    # "gpu_ms_exclusive": None if gpu_exclusive is None else float(gpu_exclusive),
                    # "gpu_ms_inclusive": None,
                    # "gpu_ms_exclusive": None,

                    "in_desc": frame.get("in_desc", []),
                    "out_desc": out_desc,

                    "param_bytes": int(frame.get("param_bytes", 0)),
                    "param_desc": frame.get("param_desc", []),
                    "weight_desc": frame.get("weight_desc", None),
                    "bias_desc": frame.get("bias_desc", None),                    
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
                    #if gpu_inclusive is not None:
                    #    parent["child_gpu_ms_sum"] += gpu_inclusive

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

    # added 02/26/2026
    ap.add_argument("--trace_enrich", action="store_true",
        help="After torch_profile export, parse trace and add gpu_ms_*_trace into layer_profile_out JSONL.")
    ap.add_argument("--layer_profile_out_enriched", type=str, default=None,
        help="Where to write enriched layer JSONL. Default: <layer_profile_out>.enriched.jsonl")    

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

                # -------------------------
                # Layer profiler (CPU incl/excl + ids + MOD:: ranges)
                # -------------------------
                prof = None
                if args.layer_profile_out:
                    prof = ExclusiveModuleProfiler(
                        model,
                        name_filter=lambda n, m: default_filter(n, m),
                        cuda_sync=args.layer_profile_cuda_sync,  # (now unused if you removed CUDA events; ok to keep)
                        max_records=200000,
                    )
                    # reset IDs per clip
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

                prof_torch = None
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

                # -------------------------
                # Stop profiler + write layer JSONL FIRST
                # -------------------------
                layer_jsonl_path = args.layer_profile_out
                if prof is not None:
                    prof.stop()

                    # write JSONL (append per clip)
                    with open(layer_jsonl_path, "a") as f:
                        for r in prof.records:
                            r["clip_id"] = cid
                            f.write(json.dumps(r) + "\n")

                # -------------------------
                # Enrich from trace AFTER layer JSONL exists
                # -------------------------
                enriched_path = None
                if args.torch_profile and args.trace_enrich and layer_jsonl_path:
                    enriched_path = args.layer_profile_out_enriched
                    if enriched_path is None:
                        enriched_path = layer_jsonl_path + ".enriched.jsonl"

                    enrich_jsonl_with_trace_gpu(
                        trace_path=args.torch_profile_out,
                        layers_jsonl_in=layer_jsonl_path,
                        layers_jsonl_out=enriched_path,
                        compute_exclusive=True,
                    )

                # -------------------------
                # Print Top-30 by trace GPU exclusive (if available)
                # -------------------------
                if prof is not None:
                    # If we enriched, read those numbers back and join by (gid, mid)
                    trace_gpu = {}
                    if enriched_path and os.path.exists(enriched_path):
                        with open(enriched_path, "r") as f:
                            for line in f:
                                try:
                                    rr = json.loads(line)
                                except Exception:
                                    continue
                                if rr.get("clip_id") != cid:
                                    continue
                                gid = rr.get("global_call_id")
                                mid = rr.get("module_call_idx")
                                if isinstance(gid, int) and isinstance(mid, int):
                                    trace_gpu[(gid, mid)] = (
                                        float(rr.get("gpu_ms_exclusive_trace", 0.0)),
                                        float(rr.get("gpu_ms_inclusive_trace", 0.0)),
                                    )

                    # Build list for printing
                    rows = []
                    for r in prof.records:
                        gid = r.get("global_call_id")
                        mid = r.get("module_call_idx")
                        gex, gin = (0.0, 0.0)
                        if isinstance(gid, int) and isinstance(mid, int):
                            gex, gin = trace_gpu.get((gid, mid), (0.0, 0.0))
                        rows.append((gex, gin, r))

                    rows.sort(key=lambda x: x[0], reverse=True)

                    print("\n[LayerProfile] Top 30 modules by gpu_ms_exclusive_trace:")
                    for gex, gin, r in rows[:30]:
                        print(
                            f"{gex:9.3f} ms (incl {gin:9.3f} ms)  "
                            f'{r["in_bytes"]/1e6:8.1f}MB -> {r["out_bytes"]/1e6:8.1f}MB  '
                            f'{r["type"]:24s}  {r["module"]}#gid={r.get("global_call_id")}#mid={r.get("module_call_idx")}'
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


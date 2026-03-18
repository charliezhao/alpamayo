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
import multiprocessing as mp
from parent_trace_helper import install_parent_trace

# =========================
# Optional faster JSON
# =========================
try:
    import orjson
    _HAS_ORJSON = True
except Exception:
    _HAS_ORJSON = False


# =========================
# Async JSONL Writer (workers)
# =========================
def _writer_worker(q: mp.Queue, out_path: str, use_orjson: bool, flush_every: int = 2048):
    """
    Worker process: consumes dict records, serializes, writes JSONL.
    """
    # Binary mode allows orjson (bytes) and normal json (encode)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "ab", buffering=1024 * 1024) as f:
        n = 0
        while True:
            item = q.get()
            if item is None:
                break
            try:
                if use_orjson:
                    f.write(orjson.dumps(item))
                    f.write(b"\n")
                else:
                    f.write((json.dumps(item) + "\n").encode("utf-8"))
            except Exception:
                # last resort: try stringifying
                try:
                    f.write((json.dumps({"_writer_error": True, "repr": repr(item)}) + "\n").encode("utf-8"))
                except Exception:
                    pass

            n += 1
            if (n % flush_every) == 0:
                f.flush()


class AsyncJSONLWriter:
    """
    Spawned worker processes to write JSONL without blocking model forward thread.
    """
    def __init__(self, out_path: str, num_workers: int = 2, max_queue: int = 50000, flush_every: int = 2048):
        # spawn is safer with CUDA than fork
        ctx = mp.get_context("spawn")
        self.q = ctx.Queue(maxsize=max_queue)
        self.ps = []
        self.out_path = out_path
        self.use_orjson = _HAS_ORJSON
        self.flush_every = flush_every

        for _ in range(max(1, int(num_workers))):
            p = ctx.Process(
                target=_writer_worker,
                args=(self.q, out_path, self.use_orjson, flush_every),
                daemon=True,
            )
            p.start()
            self.ps.append(p)

    def put(self, rec: dict):
        # Blocking put is okay; if this blocks, you’re still producing too much.
        self.q.put(rec)

    def close(self):
        for _ in self.ps:
            self.q.put(None)
        for p in self.ps:
            p.join()


# =========================
# Trace parsing utilities (unchanged)
# =========================
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
    m = _MOD_RE.match(name)
    if not m:
        return None
    return m.group("mod"), int(m.group("gid")), int(m.group("mid"))

def _is_cuda_kernel_event(ev):
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

    hints = ("nvjet_", "cublas", "cudnn", "cutlass", "triton", "flash", "vectorized_elementwise", "void ")
    return name.startswith(hints) or any(h in name for h in hints)


def enrich_jsonl_with_trace_gpu(
    trace_path: str,
    layers_jsonl_in: str,
    layers_jsonl_out: str,
    compute_exclusive: bool = True,
):
    events = _load_chrome_trace_events(trace_path)

    # 1) Collect GPU MOD slices keyed by (gid, mid)
    mod_gpu = {}
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
            f"No GPU MOD::... slices found in trace (gpu_user_annotation): {trace_path}"
        )

    # 2) Collect CUDA kernel events with pid/tid
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

    def _same_track(k, mod):
        if mod.get("pid", None) is None or mod.get("tid", None) is None:
            return True
        return (k.get("pid", None) == mod["pid"]) and (k.get("tid", None) == mod["tid"])

    # 3) Inclusive GPU time: kernel START inside [ts, te)
    gpu_incl_ms = defaultdict(float)
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
            s_us += k["dur_us"]
        gpu_incl_ms[key] = s_us / 1000.0

    # 4) Exclusive GPU time: nesting by interval containment
    gpu_excl_ms = {}
    if compute_exclusive:
        groups = defaultdict(list)
        for key, itv in mod_gpu.items():
            groups[(itv.get("pid", None), itv.get("tid", None))].append((key, itv["ts"], itv["te"]))

        children = defaultdict(list)
        for _, items in groups.items():
            items.sort(key=lambda x: (x[1], -(x[2]-x[1])))
            stack = []
            for key, ts, te in items:
                while stack and not (stack[-1][1] <= ts and te <= stack[-1][2]):
                    stack.pop()
                if stack:
                    children[stack[-1][0]].append(key)
                stack.append((key, ts, te))

        for key in mod_gpu.keys():
            incl = gpu_incl_ms.get(key, 0.0)
            child_sum = sum(gpu_incl_ms.get(ck, 0.0) for ck in children.get(key, []))
            excl = incl - child_sum
            gpu_excl_ms[key] = max(0.0, excl)

    # 5) Write enriched JSONL
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


# =========================
# Layer profile utilities (unchanged, but cached)
# =========================
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
    t = (module_type or "").lower()
    m = (module_name or "").lower()
    in0 = _first_desc(in_desc)
    out0 = _first_desc(out_desc)

    if t == "linear":
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
        return True, "conv", None

    if t == "embedding":
        return False, "embedding_gather", None

    if "attention" in t or ".attn" in m:
        return True, "attn", None

    if any(k in t for k in ["gelu", "silu", "relu", "tanh", "sigmoid"]):
        return False, "elementwise", None

    return False, "unknown", None

def default_filter(name: str, mod: torch.nn.Module) -> bool:
    n = name.lower()
    t = type(mod).__name__.lower()
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
    Producer: forward hooks build record dicts.
    Writer: async JSONL workers handle serialization + file writes.
    """
    def __init__(self, model, writer: AsyncJSONLWriter, name_filter=None, max_records=200000):
        self.model = model
        self.writer = writer
        self.name_filter = name_filter
        self.max_records = max_records
        self._handles = []
        self.use_cuda = torch.cuda.is_available()
        self._stack = []
        self._global_call_id = 0
        self._per_module_call_id = defaultdict(int)
        self.records = []  # keep for in-process post (Top-30 printing). You can disable if too big.
        self.param_cache = {}  # module_name -> (p_desc, p_bytes, w_desc, b_desc)

    def _should_keep(self, name, mod):
        if name == "":
            return False
        if self.name_filter is None:
            return True
        return self.name_filter(name, mod)

    def _build_param_cache(self):
        self.param_cache.clear()
        for name, mod in self.model.named_modules():
            if not self._should_keep(name, mod):
                continue
            try:
                self.param_cache[name] = _summarize_params(mod)
            except Exception:
                self.param_cache[name] = ([], 0, None, None)

    def reset_ids(self):
        self._global_call_id = 0
        self._per_module_call_id.clear()

    def start(self):
        self._build_param_cache()

        for name, mod in self.model.named_modules():
            if not self._should_keep(name, mod):
                continue

            def pre_hook(_mod, inputs, _name=name):
                if len(self.records) >= self.max_records:
                    return

                self._global_call_id += 1
                gid = self._global_call_id

                mid = self._per_module_call_id[_name]
                self._per_module_call_id[_name] += 1

                frame = {
                    "module": _name,
                    "type": type(_mod).__name__,
                    "global_call_id": gid,
                    "module_call_idx": mid,
                    "t0": time.perf_counter(),
                    "child_cpu_ms_sum": 0.0,
                }

                # cached param summary (big win)
                p_desc, p_bytes, w_desc, b_desc = self.param_cache.get(_name, ([], 0, None, None))
                frame["param_desc"] = p_desc
                frame["param_bytes"] = int(p_bytes)
                frame["weight_desc"] = w_desc
                frame["bias_desc"] = b_desc

                # torch.profiler range
                rf = record_function(f"MOD::{_name}#gid={gid}#mid={mid}")
                rf.__enter__()
                frame["rf"] = rf

                # input summary (still computed here)
                in_desc, in_bytes = _summarize_io(inputs)
                frame["in_desc"] = in_desc
                frame["in_bytes"] = int(in_bytes)

                self._stack.append(frame)

            def post_hook(_mod, inputs, outputs, _name=name):
                if not self._stack:
                    return
                frame = self._stack.pop()
                t1 = time.perf_counter()

                out_desc, out_bytes = _summarize_io(outputs)

                cpu_inclusive = (t1 - frame["t0"]) * 1000.0
                cpu_exclusive = cpu_inclusive - frame["child_cpu_ms_sum"]
                if cpu_exclusive < 0:
                    cpu_exclusive = 0.0

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

                    "in_desc": frame.get("in_desc", []),
                    "out_desc": out_desc,

                    "param_bytes": int(frame.get("param_bytes", 0)),
                    "param_desc": frame.get("param_desc", []),
                    "weight_desc": frame.get("weight_desc", None),
                    "bias_desc": frame.get("bias_desc", None),

                    "global_call_id": frame["global_call_id"],
                    "module_call_idx": frame["module_call_idx"],
                }

                mac_like, mac_kind, est_macs = infer_mac_fields(
                    module_type=rec["type"],
                    module_name=rec["module"],
                    in_desc=rec["in_desc"],
                    out_desc=rec["out_desc"],
                )
                rec["mac_like"] = mac_like
                rec["mac_kind"] = mac_kind
                rec["est_macs"] = est_macs

                # Producer -> queue (worker writes)
                self.writer.put(rec)

                # Keep locally for Top-30 printing
                self.records.append(rec)

                # accumulate to parent for exclusive
                if self._stack:
                    parent = self._stack[-1]
                    parent["child_cpu_ms_sum"] += cpu_inclusive

            self._handles.append(mod.register_forward_pre_hook(pre_hook))
            self._handles.append(mod.register_forward_hook(post_hook))

    def stop(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        self._stack = []


# =========================
# Ensure src/ layout works
# =========================
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper
import physical_ai_av


# =========================
# Utilities
# =========================
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


# =========================
# Main
# =========================
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
    ap.add_argument("--model_path", type=str, default="nvidia/Alpamayo-R1-10B")

    # layer profile
    ap.add_argument("--layer_profile_out", type=str, default=None,
                    help="If set, write per-module timing+IO+params info to this JSONL (async writer).")
    ap.add_argument("--layer_profile_workers", type=int, default=2,
                    help="Number of writer worker processes for layer_profile_out.")
    ap.add_argument("--layer_profile_queue", type=int, default=50000,
                    help="Max queue size for layer_profile_out (bigger uses more RAM).")
    ap.add_argument("--layer_profile_max_records", type=int, default=200000)

    # torch profiler
    ap.add_argument("--torch_profile", action="store_true")
    ap.add_argument("--torch_profile_out", type=str, default="../alpamayo_outputs/torch_trace.json")
    ap.add_argument("--trace_enrich", action="store_true")
    ap.add_argument("--layer_profile_out_enriched", type=str, default=None)

    # 20260306
    ap.add_argument(
        "--visual_trace_out",
        type=str,
        default="",
        help="If set, record a dedicated trace of vlm.model.visual forward into this directory.",
    )

    args = ap.parse_args()

    host = socket.gethostname()
    torch_version = torch.__version__
    cuda_version = torch.version.cuda
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    amp_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    model_dtype = amp_dtype

    with open(args.clip_ids_file) as f:
        clip_ids = [ln.strip() for ln in f if ln.strip()]
    clip_ids = clip_ids[: args.max_clips]

    processed_clips = set()
    if os.path.exists(args.out_jsonl):
        with open(args.out_jsonl, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_clips.add(data['clip_id'])
                except Exception:
                    continue
    print(f"Skipping {len(processed_clips)} already processed clips.")

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    """
    # Model load (your hack preserved)
    model = AlpamayoR1.from_pretrained(
        args.model_path,
        dtype=model_dtype,
        ignore_mismatched_sizes=True,
    )

    
    # 20260306. install the visual trace helper
    from visual_trace_helper import install_visual_trace

    if args.visual_trace_out:
        visual_trace_recorder = install_visual_trace(
            model=model,
            out_dir=args.visual_trace_out,
            enable_once=True
        )
        print(f"[visual-trace] installed -> {args.visual_trace_out}")
    #end of visual_trace_helper
    """
    """
    # 20260306. install the parent trace helper
    parent_trace_recorder = install_parent_trace(
        model=model,
        out_dir="/outputs/parent_trace",
        enable_once=True,
    )
    print("[parent-trace] installed -> /outputs/parent_trace")  
    """  
    """
    model.to_empty(device=args.device)
    model.to(args.device)
    """

    model = AlpamayoR1.from_pretrained(
        args.model_path,
        dtype=model_dtype,
    ).to(args.device)

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
        trajectories, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=args.top_p,
            temperature=args.temperature,
            num_traj_samples=args.num_traj_samples,
            max_generation_length=args.max_generation_length,
            return_extra=False,
        )
    torch.cuda.synchronize()

    # Convert predicted trajectories to JSON serializable format
    traj_list = None
    if trajectories is not None:
        traj_list = trajectories.detach().cpu().tolist()    

    # -------------------------
    # Layer-profile async writer (global)
    # -------------------------
    layer_writer = None
    if args.layer_profile_out:
        # Start workers once for the whole run
        layer_writer = AsyncJSONLWriter(
            out_path=args.layer_profile_out,
            num_workers=args.layer_profile_workers,
            max_queue=args.layer_profile_queue,
            flush_every=2048,
        )
        print(f"[layer_profile] async writer started: workers={args.layer_profile_workers}, "
              f"queue={args.layer_profile_queue}, out={args.layer_profile_out}, orjson={_HAS_ORJSON}")

    # -------------------------
    # Main loop
    # -------------------------
    mode = "a" if os.path.exists(args.out_jsonl) else "w"
    with open(args.out_jsonl, mode) as out:
        for cid in tqdm(clip_ids, desc="Clips"):
            if cid in processed_clips:
                continue

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
                start = torch.cuda.Event(True)
                end = torch.cuda.Event(True)
                torch.cuda.reset_peak_memory_stats()
                start.record()
                t0 = time.time()

                # -------------------------
                # Layer profiler (producer -> queue -> workers)
                # -------------------------
                prof = None
                if layer_writer is not None:
                    prof = ExclusiveModuleProfiler(
                        model,
                        writer=layer_writer,
                        name_filter=lambda n, m: default_filter(n, m),
                        max_records=args.layer_profile_max_records,
                    )
                    prof.reset_ids()
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

                # Unpack returns
                if isinstance(ret, tuple) and len(ret) == 3:
                    pred_xyz, pred_rot, extra = ret
                elif isinstance(ret, tuple) and len(ret) == 2:
                    pred_xyz, pred_rot = ret
                    extra = {}
                else:
                    raise TypeError(f"Unexpected return: type={type(ret)} value={ret!r}")

                # stop hooks (important: do this before end.record)
                if prof is not None:
                    prof.stop()

                # optional trace enrich (still uses layer_profile_out file)
                if args.torch_profile and args.trace_enrich and args.layer_profile_out:
                    enriched_path = args.layer_profile_out_enriched
                    if enriched_path is None:
                        enriched_path = args.layer_profile_out + ".enriched.jsonl"

                    # NOTE: because writing is async, you must ensure queue drains before enriching.
                    # The simplest safe approach: close and restart writer around enrich,
                    # but that’s heavy. Instead, do enrich only for 1-clip runs after program ends.
                    #
                    # We will warn here:
                    print("[trace_enrich] WARNING: layer_profile_out is written asynchronously. "
                          "For correct enrich results, run 1 clip and do enrich after the program ends.")

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

                cot = extra.get("cot", None)
                if cot is not None:
                    try:
                        rec["cot"] = cot[0]
                    except Exception:
                        rec["cot"] = cot
                else:
                    rec["cot"] = None

                rec["trajectory"] = traj_list

            except Exception:
                rec["ok"] = False
                rec["e2e_clip_s"] = time.time() - t_clip0
                rec["error"] = traceback.format_exc()

            out.write(json.dumps(json_safe(rec)) + "\n")
            out.flush()

            torch.cuda.empty_cache()

    # Drain and close writer processes
    if layer_writer is not None:
        print("[layer_profile] closing async writer (draining queue)...")
        layer_writer.close()

    print(f"Wrote JSONL: {args.out_jsonl}")


if __name__ == "__main__":
    main()
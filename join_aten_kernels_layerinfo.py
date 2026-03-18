#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# Helpers: dtype size + bytes
# -----------------------------
DTYPE_BYTES = {
    "float32": 4, "fp32": 4, "f32": 4,
    "float": 4,
    "float16": 2, "fp16": 2, "f16": 2, "half": 2,
    "bfloat16": 2, "bf16": 2,
    "int64": 8, "i64": 8, "long": 8,
    "int32": 4, "i32": 4,
    "int16": 2, "i16": 2,
    "int8": 1, "i8": 1,
    "uint8": 1, "u8": 1,
    "bool": 1,
}

def normalize_dtype(dt: Optional[str]) -> Optional[str]:
    if not dt:
        return None
    s = str(dt).strip().lower()
    s = s.replace("torch.", "")
    return s

def prod_shape(shape: Any) -> Optional[int]:
    if shape is None:
        return None
    if isinstance(shape, (list, tuple)):
        p = 1
        for x in shape:
            try:
                xi = int(x)
            except Exception:
                return None
            if xi < 0:
                return None
            p *= xi
        return p
    return None

def est_bytes(shape: Any, dtype: Optional[str]) -> Optional[int]:
    dt = normalize_dtype(dtype)
    if dt is None:
        return None
    b = DTYPE_BYTES.get(dt)
    if b is None:
        return None
    n = prod_shape(shape)
    if n is None:
        return None
    return n * b

# -----------------------------
# Load layer jsonl (schema-flexible)
# -----------------------------
def load_layer_jsonl(path: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """
    Returns:
      - by_gid_str: gid(str)->payload
      - by_rfid: record_function_id(int)->payload
    """
    by_gid: Dict[str, Dict[str, Any]] = {}
    by_rfid: Dict[int, Dict[str, Any]] = {}

    if not path or not os.path.exists(path):
        return by_gid, by_rfid

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            # common IDs
            gid = obj.get("gid", None)
            if gid is None:
                gid = obj.get("group_id", None)
            if gid is None:
                gid = obj.get("op_gid", None)

            rfid = obj.get("record_function_id", None)
            if rfid is None:
                rfid = obj.get("rf_id", None)
            if rfid is None:
                rfid = obj.get("External id", None)  # sometimes used

            if gid is not None:
                by_gid[str(gid)] = obj
            if rfid is not None:
                try:
                    by_rfid[int(rfid)] = obj
                except Exception:
                    pass

    return by_gid, by_rfid

# -----------------------------
# Load torch chrome trace json
# -----------------------------
def load_trace_events(trace_json_path: str) -> List[Dict[str, Any]]:
    with open(trace_json_path, "r", encoding="utf-8", errors="replace") as f:
        trace = json.load(f)

    # torch.profiler chrome trace usually has {"traceEvents":[...]}
    if isinstance(trace, dict) and "traceEvents" in trace and isinstance(trace["traceEvents"], list):
        return trace["traceEvents"]
    # sometimes it’s directly a list
    if isinstance(trace, list):
        return trace
    raise RuntimeError(f"Unrecognized trace json structure in {trace_json_path}")

def is_cpu_aten_event(ev: Dict[str, Any]) -> bool:
    name = str(ev.get("name", ""))
    if not name.startswith("aten::"):
        return False
    if ev.get("ph") != "X":
        return False
    # heuristics: CPU events usually have tid/pid and cat includes "cpu" or "Operator"
    return True

def is_cuda_kernel_event(ev: Dict[str, Any]) -> bool:
    if ev.get("ph") != "X":
        return False
    cat = str(ev.get("cat", "")).lower()
    name = str(ev.get("name", ""))
    # heuristics: kineto uses cat "cuda_kernel" or "Kernel"
    if "cuda" in cat and "kernel" in cat:
        return True
    if cat == "kernel":
        return True
    # some builds label kernels as their demangled name with cat empty but args has "device"
    args = ev.get("args", {}) or {}
    if "stream" in args and ("cuda" in cat or "kernel" in cat):
        return True
    return False

def get_time_us(ev: Dict[str, Any]) -> Tuple[float, float]:
    # chrome trace uses microseconds for ts and dur (usually)
    ts = float(ev.get("ts", 0.0))
    dur = float(ev.get("dur", 0.0))
    return ts, ts + dur

def try_get_id(ev: Dict[str, Any], keys: List[str]) -> Optional[int]:
    args = ev.get("args", {}) or {}
    for k in keys:
        if k in args:
            try:
                return int(args[k])
            except Exception:
                pass
        if k in ev:
            try:
                return int(ev[k])
            except Exception:
                pass
    return None

# -----------------------------
# Build mappings: kernels -> aten
# -----------------------------
def build_aten_intervals(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Returns list of aten events with:
      name, ts0, ts1, dur, external_id, corr_id, rfid
    """
    out = []
    for ev in events:
        if not is_cpu_aten_event(ev):
            continue
        ts0, ts1 = get_time_us(ev)
        if ts1 <= ts0:
            continue
        args = ev.get("args", {}) or {}
        out.append({
            "aten": ev.get("name"),
            "ts0": ts0,
            "ts1": ts1,
            "dur": ts1 - ts0,
            "pid": ev.get("pid"),
            "tid": ev.get("tid"),
            "external_id": try_get_id(ev, ["External id", "external_id", "External Id"]),
            "corr_id": try_get_id(ev, ["Correlation id", "correlation", "correlation_id"]),
            "rfid": try_get_id(ev, ["Record function id", "record_function_id", "rfid"]),
            "raw_args": args,
        })
    # sort by start then by duration (shorter first for containment)
    out.sort(key=lambda x: (x["ts0"], x["dur"]))
    return out

def build_kernel_rows(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Returns list of kernel events with:
      kernel_name, ts0, ts1, dur, stream, corr_id, external_id
    """
    out = []
    for ev in events:
        if not is_cuda_kernel_event(ev):
            continue
        ts0, ts1 = get_time_us(ev)
        if ts1 <= ts0:
            continue
        args = ev.get("args", {}) or {}
        out.append({
            "kernel": ev.get("name"),
            "ts0": ts0,
            "ts1": ts1,
            "dur_us": ts1 - ts0,
            "stream": args.get("stream", args.get("Stream", None)),
            "external_id": try_get_id(ev, ["External id", "external_id", "External Id"]),
            "corr_id": try_get_id(ev, ["Correlation id", "correlation", "correlation_id"]),
            "grid": args.get("grid", args.get("Grid", None)),
            "block": args.get("block", args.get("Block", None)),
            "raw_args": args,
        })
    out.sort(key=lambda x: x["ts0"])
    return out

def assign_kernels_to_aten(
    kernels: List[Dict[str, Any]],
    atens: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    For each kernel, assign an aten op using:
      1) ID join (external_id / corr_id) if possible
      2) time containment: smallest aten interval that contains kernel start
    """
    # Build ID maps
    by_external = defaultdict(list)
    by_corr = defaultdict(list)
    for a in atens:
        if a["external_id"] is not None:
            by_external[a["external_id"]].append(a)
        if a["corr_id"] is not None:
            by_corr[a["corr_id"]].append(a)

    # sort lists by duration (smallest first)
    for k in list(by_external.keys()):
        by_external[k].sort(key=lambda x: x["dur"])
    for k in list(by_corr.keys()):
        by_corr[k].sort(key=lambda x: x["dur"])

    # For containment fallback, we do a simple scan; for big traces you can optimize, but for 1-clip it’s fine.
    assigned = []
    for ker in kernels:
        aten = None
        # 1) ID join
        if ker["external_id"] is not None and ker["external_id"] in by_external:
            aten = by_external[ker["external_id"]][0]
        elif ker["corr_id"] is not None and ker["corr_id"] in by_corr:
            aten = by_corr[ker["corr_id"]][0]
        else:
            # 2) containment join
            t = ker["ts0"]
            best = None
            for a in atens:
                if a["ts0"] <= t <= a["ts1"]:
                    if best is None or a["dur"] < best["dur"]:
                        best = a
            aten = best

        row = dict(ker)
        if aten:
            row["aten"] = aten["aten"]
            row["aten_ts0"] = aten["ts0"]
            row["aten_ts1"] = aten["ts1"]
            row["aten_dur_us"] = aten["dur"]
            row["aten_external_id"] = aten["external_id"]
            row["aten_corr_id"] = aten["corr_id"]
            row["aten_rfid"] = aten["rfid"]
        else:
            row["aten"] = None
            row["aten_ts0"] = None
            row["aten_ts1"] = None
            row["aten_dur_us"] = None
            row["aten_external_id"] = None
            row["aten_corr_id"] = None
            row["aten_rfid"] = None
        assigned.append(row)

    return assigned

# -----------------------------
# Enrich with layer info
# -----------------------------
def enrich_with_layer(
    rows: List[Dict[str, Any]],
    by_gid: Dict[str, Dict[str, Any]],
    by_rfid: Dict[int, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        layer = None

        # Prefer gid if present on kernel args (rare), else use aten rfid/external id
        # We'll try a chain of possible keys.
        gid = None
        args = r.get("raw_args", {}) or {}
        for k in ["gid", "group_id", "op_gid"]:
            if k in args:
                gid = args.get(k)
                break

        if gid is not None and str(gid) in by_gid:
            layer = by_gid[str(gid)]
        else:
            # try record function id
            rfid = r.get("aten_rfid")
            if rfid is not None and int(rfid) in by_rfid:
                layer = by_rfid[int(rfid)]
            else:
                # fall back to external id (some logs store this)
                ext = r.get("aten_external_id")
                if ext is not None and int(ext) in by_rfid:
                    layer = by_rfid[int(ext)]

        row = dict(r)
        if layer:
            # Pull commonly desired fields if present
            row["gid"] = layer.get("gid", layer.get("group_id", layer.get("op_gid", "")))
            row["est_macs"] = layer.get("est_macs", layer.get("macs", layer.get("estimated_macs", "")))

            # Copy any *_desc fields through (in/out/param/weight/bias_desc)
            for key in ["in_desc", "out_desc", "param_desc", "weight_desc", "bias_desc",
                        "input_desc", "output_desc", "inputs_desc", "outputs_desc"]:
                if key in layer and key not in row:
                    row[key] = layer.get(key)

            # If layer stores shapes/dtypes separately, bring them over
            for key in ["in_shape", "out_shape", "weight_shape", "bias_shape",
                        "in_dtype", "out_dtype", "weight_dtype", "bias_dtype"]:
                if key in layer and key not in row:
                    row[key] = layer.get(key)

            # Compute bytes if possible
            in_shape = row.get("in_shape", None)
            in_dtype = row.get("in_dtype", None)
            out_shape = row.get("out_shape", None)
            out_dtype = row.get("out_dtype", None)
            w_shape = row.get("weight_shape", None)
            w_dtype = row.get("weight_dtype", None)
            b_shape = row.get("bias_shape", None)
            b_dtype = row.get("bias_dtype", None)

            row["in_bytes_est"] = est_bytes(in_shape, in_dtype)
            row["out_bytes_est"] = est_bytes(out_shape, out_dtype)
            row["weight_bytes_est"] = est_bytes(w_shape, w_dtype)
            row["bias_bytes_est"] = est_bytes(b_shape, b_dtype)
        else:
            row["gid"] = ""
            row["est_macs"] = ""
            row["in_desc"] = ""
            row["out_desc"] = ""
            row["param_desc"] = ""
            row["weight_desc"] = ""
            row["bias_desc"] = ""
            row["in_bytes_est"] = ""
            row["out_bytes_est"] = ""
            row["weight_bytes_est"] = ""
            row["bias_bytes_est"] = ""

        out.append(row)
    return out

# -----------------------------
# Output CSV
# -----------------------------
def write_csv(rows: List[Dict[str, Any]], out_csv: str, aggregate: bool):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    if aggregate:
        # aggregate by (aten, kernel, gid)
        agg = {}
        for r in rows:
            key = (r.get("aten") or "", r.get("kernel") or "", str(r.get("gid") or ""))
            if key not in agg:
                agg[key] = dict(r)
                agg[key]["kernel_calls"] = 0
                agg[key]["kernel_total_cuda_us"] = 0.0
                agg[key]["kernel_avg_cuda_us"] = 0.0
            agg[key]["kernel_calls"] += 1
            agg[key]["kernel_total_cuda_us"] += float(r.get("dur_us", 0.0))

        for v in agg.values():
            c = v["kernel_calls"]
            v["kernel_avg_cuda_us"] = (v["kernel_total_cuda_us"] / c) if c else 0.0

        out_rows = sorted(agg.values(), key=lambda x: x.get("kernel_total_cuda_us", 0.0), reverse=True)
    else:
        out_rows = rows

    # choose columns (stable + extra)
    base_cols = [
        "aten",
        "kernel",
        "dur_us",
        "ts0",
        "ts1",
        "stream",
        "grid",
        "block",
        "gid",
        "est_macs",
        "in_desc",
        "out_desc",
        "param_desc",
        "weight_desc",
        "bias_desc",
        "in_shape",
        "in_dtype",
        "in_bytes_est",
        "out_shape",
        "out_dtype",
        "out_bytes_est",
        "weight_shape",
        "weight_dtype",
        "weight_bytes_est",
        "bias_shape",
        "bias_dtype",
        "bias_bytes_est",
        "aten_external_id",
        "aten_corr_id",
        "aten_rfid",
        "aten_ts0",
        "aten_ts1",
        "aten_dur_us",
    ]
    if aggregate:
        base_cols = ["kernel_calls", "kernel_total_cuda_us", "kernel_avg_cuda_us"] + base_cols

    # add any additional keys we discovered
    extra_keys = []
    seen = set(base_cols)
    for r in out_rows:
        for k in r.keys():
            if k not in seen and k != "raw_args":
                extra_keys.append(k)
                seen.add(k)

    cols = base_cols + extra_keys

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            rr = dict(r)
            # remove huge blobs
            rr.pop("raw_args", None)
            w.writerow(rr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_json", required=True, help="torch profiler chrome trace json (export_chrome_trace output)")
    ap.add_argument("--layer_jsonl", default="", help="layer profiling jsonl (optional)")
    ap.add_argument("--out_csv", required=True, help="output CSV path")
    ap.add_argument("--aggregate", action="store_true", help="aggregate by (aten,kernel,gid) instead of per kernel instance")
    args = ap.parse_args()

    events = load_trace_events(args.trace_json)
    atens = build_aten_intervals(events)
    kernels = build_kernel_rows(events)

    assigned = assign_kernels_to_aten(kernels, atens)

    by_gid, by_rfid = load_layer_jsonl(args.layer_jsonl) if args.layer_jsonl else ({}, {})
    enriched = enrich_with_layer(assigned, by_gid, by_rfid)

    write_csv(enriched, args.out_csv, aggregate=args.aggregate)
    print(f"Wrote: {args.out_csv}")
    print(f"Rows: {len(enriched)}  (aggregate={args.aggregate})")
    print("Note: aten↔kernel mapping uses correlation IDs when available, else time-window containment.")

if __name__ == "__main__":
    main()
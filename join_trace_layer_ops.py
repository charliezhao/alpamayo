#!/usr/bin/env python3
import argparse, json, csv, os, re, heapq
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

# MOD scope name format:  MOD::...#gid=3#mid=0
RX_GID = re.compile(r"#gid=(\d+)#", re.I)

def load_trace_events(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        trace = json.load(f)
    if isinstance(trace, dict) and "traceEvents" in trace:
        return trace["traceEvents"]
    if isinstance(trace, list):
        return trace
    raise RuntimeError("Unrecognized trace json format")

def load_layer_jsonl(path: str) -> Dict[int, Dict[str, Any]]:
    by_gcid = {}
    rows = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            obj=json.loads(line)
            rows += 1
            gcid=obj.get("global_call_id", None)
            if gcid is None:
                continue
            by_gcid[int(gcid)] = obj
    print(f"[layer] rows={rows} indexed_by_global_call_id={len(by_gcid)}")
    return by_gcid

def is_complete(ev: Dict[str, Any]) -> bool:
    return ev.get("ph") == "X" and "ts" in ev and "dur" in ev

def get_time(ev: Dict[str, Any]) -> Tuple[float, float]:
    ts=float(ev.get("ts", 0.0))
    dur=float(ev.get("dur", 0.0))
    return ts, ts+dur

def is_mod_scope(ev: Dict[str, Any]) -> bool:
    if not is_complete(ev):
        return False
    if str(ev.get("cat","")) != "user_annotation":
        return False
    name=str(ev.get("name",""))
    return name.startswith("MOD::") and ("#gid=" in name)

def extract_gid_from_mod(ev: Dict[str, Any]) -> Optional[int]:
    m = RX_GID.search(str(ev.get("name","")))
    if not m:
        return None
    return int(m.group(1))

def is_cpu_aten(ev: Dict[str, Any]) -> bool:
    return is_complete(ev) and str(ev.get("cat","")) == "cpu_op" and str(ev.get("name","")).startswith("aten::")

# ---------- Robust kernel detection ----------
# In torch profiler JSON, CUDA kernels often appear as cat="cuda_kernel" (not "kernel").
# Sometimes cat includes "cuda" and name looks like "void at::native::..." or "nvjet_tst_..."
def is_cuda_kernel(ev: Dict[str, Any]) -> bool:
    if not is_complete(ev):
        return False
    cat=str(ev.get("cat","")).lower()

    # common kineto cats
    if "cuda_kernel" in cat:
        return True
    if cat == "kernel":
        return True

    # fallback heuristic: CUDA-ish categories AND kernel-like names
    if "cuda" in cat or "gpu" in cat:
        name=str(ev.get("name",""))
        if name.startswith("void ") or "kernel" in name or name.startswith("nvjet_") or "cutlass" in name or "cublas" in name:
            return True

    return False

def cat_hist(events: List[Dict[str, Any]], topn=25):
    c=Counter()
    for ev in events:
        c[str(ev.get("cat",""))] += 1
    print("[trace] top cats:")
    for k,v in c.most_common(topn):
        print(f"  {k:20s} {v}")

def build_intervals_mod(events: List[Dict[str, Any]]):
    intervals=[]
    for ev in events:
        if not is_mod_scope(ev):
            continue
        gid=extract_gid_from_mod(ev)
        if gid is None:
            continue
        ts0, ts1 = get_time(ev)
        if ts1 <= ts0:
            continue
        intervals.append((ts0, ts1, ts1-ts0, gid, ev.get("name","")))
    intervals.sort(key=lambda x: x[0])
    print(f"[trace] MOD intervals: {len(intervals)}")
    return intervals

def build_intervals_aten(events: List[Dict[str, Any]]):
    intervals=[]
    by_extid={}
    for ev in events:
        if not is_cpu_aten(ev):
            continue
        ts0, ts1 = get_time(ev)
        if ts1 <= ts0:
            continue
        args=ev.get("args",{}) or {}
        ext=args.get("External id", None)
        payload={"aten": ev.get("name",""), "external_id": ext}
        intervals.append((ts0, ts1, ts1-ts0, payload))
        if ext is not None:
            by_extid[int(ext)] = payload | {"ts0": ts0, "ts1": ts1}
    intervals.sort(key=lambda x: x[0])
    print(f"[trace] aten intervals: {len(intervals)} (with External id map size={len(by_extid)})")
    return intervals, by_extid

def assign_smallest_enclosing(points: List[Tuple[float, Dict[str, Any]]], intervals, field: str):
    """
    intervals: list of (start,end,dur,payload...)
    choose active interval with smallest dur.
    """
    i=0
    heap=[]  # (dur, end, payload_tuple)
    for t,row in points:
        while i < len(intervals) and intervals[i][0] <= t:
            start,end,dur,*payload = intervals[i]
            heapq.heappush(heap, (dur, end, payload))
            i += 1
        while heap and heap[0][1] < t:
            heapq.heappop(heap)
        if heap:
            payload = heap[0][2]
            # payload can be [gid,name] or [dict]
            if field == "gid":
                gid = payload[0]
                name = payload[1] if len(payload) > 1 else ""
                row["gid"] = gid
                row["gid_scope_name"] = name
            else:
                row[field] = payload[0].get(field, None)
        else:
            if field == "gid":
                row["gid"]=None
                row["gid_scope_name"]=""
            else:
                row[field]=None

import pandas as pd

def run_aggregation(csv_path):

    print(f"[aggregate] loading {csv_path}")

    df = pd.read_csv(csv_path)

    df["cuda_ms"] = df["kernel_cuda_us"] / 1000.0

    base = os.path.splitext(csv_path)[0]

    out_kernels = base + ".agg_top_kernels.csv"
    out_aten = base + ".agg_top_aten.csv"
    out_modules = base + ".agg_top_modules.csv"
    out_gid = base + ".agg_top_gid.csv"

    # ----------------------
    # kernels
    # ----------------------

    top_kernels = (
        df.groupby("kernel")
        .agg(time_ms=("cuda_ms","sum"),
             calls=("kernel","count"))
        .sort_values("time_ms", ascending=False)
    )

    top_kernels.to_csv(out_kernels)

    # ----------------------
    # aten ops
    # ----------------------

    top_aten = (
        df.groupby("aten")
        .agg(time_ms=("cuda_ms","sum"),
             calls=("aten","count"))
        .sort_values("time_ms", ascending=False)
    )

    top_aten.to_csv(out_aten)

    # ----------------------
    # modules
    # ----------------------

    top_modules = (
        df.groupby("module")
        .agg(time_ms=("cuda_ms","sum"),
             calls=("module","count"))
        .sort_values("time_ms", ascending=False)
    )

    top_modules.to_csv(out_modules)

    # ----------------------
    # gid layers
    # ----------------------

    top_gid = (
        df.groupby(["gid","module"])
        .agg(time_ms=("cuda_ms","sum"),
             calls=("kernel","count"))
        .sort_values("time_ms", ascending=False)
    )

    top_gid.to_csv(out_gid)

    total = df["cuda_ms"].sum()

    gemm = df[
        df["aten"].str.contains("mm|matmul|linear", na=False)
    ]["cuda_ms"].sum()

    print(f"[aggregate] total CUDA time: {total:.2f} ms")
    print(f"[aggregate] GEMM operators: {gemm:.2f} ms ({100*gemm/total:.1f}%)")

    print("[aggregate] wrote:")
    print(" ", out_kernels)
    print(" ", out_aten)
    print(" ", out_modules)
    print(" ", out_gid)    

def main():
    ap=argparse.ArgumentParser()

    ap.add_argument(
        "--trace_json",
        required=True,
        help="torch profiler chrome trace json"
    )

    ap.add_argument(
        "--layer_jsonl",
        required=True,
        help="layer profiling jsonl"
    )

    ap.add_argument(
        "--out_csv",
        required=True,
        help="output joined csv"
    )

    args=ap.parse_args()

    # normal join path
    missing = [k for k in ["trace_json","layer_jsonl","out_csv"] if getattr(args, k) is None]
    if missing:
        raise SystemExit("ERROR: join mode requires: --trace_json --layer_jsonl --out_csv")

    layers=load_layer_jsonl(args.layer_jsonl)
    events=load_trace_events(args.trace_json)
    print(f"[trace] events={len(events)}")
    cat_hist(events)

    mod_intervals=build_intervals_mod(events)
    aten_intervals, aten_by_extid = build_intervals_aten(events)

    # Collect kernels
    kernels=[]
    for ev in events:
        if not is_cuda_kernel(ev):
            continue
        ts0, ts1 = get_time(ev)
        kargs=ev.get("args",{}) or {}
        kernels.append({
            "kernel": ev.get("name",""),
            "kernel_cuda_us": ts1-ts0,
            "kernel_ts0": ts0,
            "kernel_ts1": ts1,
            "cat": ev.get("cat",""),
            "pid": ev.get("pid",""),
            "tid": ev.get("tid",""),
            "External_id_kernel": kargs.get("External id", kargs.get("External Id", "")),
            "Correlation_id": kargs.get("Correlation id", kargs.get("correlation", "")),
            "stream": kargs.get("stream", kargs.get("Stream","")),
            "grid": kargs.get("grid", kargs.get("Grid","")),
            "block": kargs.get("block", kargs.get("Block","")),
        })

    kernels.sort(key=lambda r: r["kernel_ts0"])
    print(f"[trace] detected kernels={len(kernels)}")

    # Assign gid by MOD scope containment (by time)
    points=[(r["kernel_ts0"], r) for r in kernels]
    assign_smallest_enclosing(points, mod_intervals, "gid")

    # Assign aten:
    # Prefer External id correlation if kernel provides it; else time-containment.
    # Many traces don't give kernel->ExternalId; then fallback to time is fine.
    for r in kernels:
        ext = r.get("External_id_kernel", "")
        aten_name=None
        if ext != "":
            try:
                exti=int(ext)
                if exti in aten_by_extid:
                    aten_name=aten_by_extid[exti]["aten"]
            except Exception:
                pass
        r["aten"] = aten_name  # may be None

    # Fallback: if aten still None, do time containment
    missing = [r for r in kernels if not r.get("aten")]
    if missing:
        pts=[(r["kernel_ts0"], r) for r in missing]
        assign_smallest_enclosing(pts, aten_intervals, "aten")

    # Enrich from layers by gid == global_call_id
    joined=0
    for r in kernels:
        gid=r.get("gid", None)
        layer=layers.get(int(gid)) if gid is not None else None
        if layer:
            joined += 1
            r["global_call_id_layer"] = layer.get("global_call_id","")
            r["module"] = layer.get("module","")
            r["module_call_idx"] = layer.get("module_call_idx","")
            r["type"] = layer.get("type","")
            r["est_macs"] = layer.get("est_macs","")
            r["mac_kind"] = layer.get("mac_kind","")
            r["mac_like"] = layer.get("mac_like","")

            r["in_bytes"] = layer.get("in_bytes","")
            r["out_bytes"] = layer.get("out_bytes","")
            r["param_bytes"] = layer.get("param_bytes","")

            r["in_desc"] = json.dumps(layer.get("in_desc",""), ensure_ascii=False)
            r["out_desc"] = json.dumps(layer.get("out_desc",""), ensure_ascii=False)
            r["param_desc"] = json.dumps(layer.get("param_desc",""), ensure_ascii=False)
            r["weight_desc"] = json.dumps(layer.get("weight_desc",""), ensure_ascii=False)
            r["bias_desc"] = json.dumps(layer.get("bias_desc",""), ensure_ascii=False)

            r["cpu_ms_exclusive"] = layer.get("cpu_ms_exclusive","")
            r["cpu_ms_inclusive"] = layer.get("cpu_ms_inclusive","")
        else:
            r["global_call_id_layer"] = ""

    print(f"[join] kernels_with_gid={sum(1 for r in kernels if r.get('gid') is not None)}/{len(kernels)}")
    print(f"[join] kernels_with_layer_match={joined}/{len(kernels)}")
    print(f"[join] kernels_with_aten={sum(1 for r in kernels if r.get('aten'))}/{len(kernels)}")

    # Write CSV
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    cols=[
        "gid","global_call_id_layer","gid_scope_name",
        "module","module_call_idx","type",
        "aten",
        "kernel","kernel_cuda_us","kernel_ts0","kernel_ts1",
        "cat","pid","tid","stream","grid","block",
        "External_id_kernel","Correlation_id",
        "est_macs","mac_kind","mac_like",
        "in_bytes","out_bytes","param_bytes",
        "cpu_ms_exclusive","cpu_ms_inclusive",
        "in_desc","out_desc","param_desc","weight_desc","bias_desc",
    ]

    # Filter model-only kernels
    kernels_scoped = [r for r in kernels if r.get("gid") is not None]

    print(f"[write] writing {len(kernels_scoped)} scoped kernels (out of {len(kernels)})")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in kernels_scoped:
            w.writerow(r)  

    print("Wrote:", args.out_csv)
    # run aggregation automatically
    run_aggregation(args.out_csv)

if __name__ == "__main__":
    main()



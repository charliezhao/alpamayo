import csv
import json
from pathlib import Path

BASE = Path("/outputs")

JOIN_CSV = BASE / "aten_kernel_layer_join_full_20260306.csv"
VIS_DIR = BASE / "visual_trace_20260306"
VIS_FWD = VIS_DIR / "visual_forward_debug.jsonl"
VIS_LAYERS = VIS_DIR / "visual_layers.jsonl"
VIS_CHROME = VIS_DIR / "visual_trace_chrome.json"

OUT_CSV = BASE / "stitched_visual_timeline_aligned.csv"


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


# -------------------------------------------------
# Load forward debug and find anchors
# -------------------------------------------------
fwd_rows = load_jsonl(VIS_FWD)

perf_begin = None
perf_end = None

for r in fwd_rows:
    if r.get("event") == "visual_anchor_begin":
        perf_begin = r.get("ts_perf_ns")
    elif r.get("event") == "visual_anchor_end":
        perf_end = r.get("ts_perf_ns")

if perf_begin is None or perf_end is None:
    raise RuntimeError("Could not find perf anchors in visual_forward_debug.jsonl")


# -------------------------------------------------
# Load chrome trace and find anchors
# -------------------------------------------------
with open(VIS_CHROME, "r", encoding="utf-8") as f:
    chrome = json.load(f)

trace_events = chrome.get("traceEvents", [])

chrome_begin = None
chrome_end = None

for ev in trace_events:
    name = ev.get("name")
    ts = ev.get("ts")
    if name == "VISUAL_ANCHOR_BEGIN" and ts is not None:
        chrome_begin = float(ts)
    elif name == "VISUAL_ANCHOR_END" and ts is not None:
        chrome_end = float(ts)

if chrome_begin is None or chrome_end is None:
    raise RuntimeError("Could not find chrome anchors in visual_trace_chrome.json")


# -------------------------------------------------
# Build affine transform:
# chrome_ts_us = a * perf_ns + b
# -------------------------------------------------
a = (chrome_end - chrome_begin) / (perf_end - perf_begin)
b = chrome_begin - a * perf_begin

print("Alignment solved:")
print(f"  perf_begin_ns={perf_begin}")
print(f"  perf_end_ns  ={perf_end}")
print(f"  chrome_begin_us={chrome_begin}")
print(f"  chrome_end_us  ={chrome_end}")
print(f"  a={a}")
print(f"  b={b}")


def perf_ns_to_chrome_us(ts_perf_ns):
    return a * ts_perf_ns + b


events = []

# -------------------------------------------------
# 1. Global join CSV
# -------------------------------------------------
with open(JOIN_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        ts = safe_float(r.get("kernel_ts0"))
        if ts is None:
            continue
        events.append({
            "ts_us": ts,
            "source": "kernel_join",
            "kind": "kernel",
            "name": r.get("aten") or r.get("kernel"),
            "module": r.get("module"),
            "dur_us": safe_float(r.get("kernel_cuda_us")),
            "details": json.dumps({
                "gid": r.get("gid"),
                "call_id": r.get("global_call_id_layer"),
            }, ensure_ascii=False),
        })

# -------------------------------------------------
# 2. Visual forward debug rows (aligned)
# -------------------------------------------------
for r in fwd_rows:
    ts_perf_ns = r.get("ts_perf_ns")
    if ts_perf_ns is None:
        continue

    events.append({
        "ts_us": perf_ns_to_chrome_us(ts_perf_ns),
        "source": "visual_forward",
        "kind": r.get("event"),
        "name": "vlm.model.visual.forward",
        "module": "",
        "dur_us": None,
        "details": json.dumps(r, ensure_ascii=False),
    })

# -------------------------------------------------
# 3. Visual layer rows (aligned)
# -------------------------------------------------
layer_rows = load_jsonl(VIS_LAYERS)

for r in layer_rows:
    ts_perf_ns = r.get("ts_perf_ns")
    if ts_perf_ns is None:
        continue

    wall_ms = r.get("wall_ms")
    dur_us = wall_ms * 1000.0 if wall_ms is not None else None

    events.append({
        "ts_us": perf_ns_to_chrome_us(ts_perf_ns),
        "source": "visual_layer",
        "kind": r.get("event"),
        "name": r.get("module"),
        "module": r.get("type"),
        "dur_us": dur_us,
        "details": json.dumps({
            "call_idx": r.get("module_call_idx"),
            "in_desc": r.get("in_desc"),
            "out_desc": r.get("out_desc"),
        }, ensure_ascii=False),
    })

# -------------------------------------------------
# 4. Chrome trace rows
# -------------------------------------------------
for ev in trace_events:
    name = ev.get("name")
    ts = ev.get("ts")
    if not name or ts is None:
        continue

    if not (
        str(name).startswith("aten::")
        or str(name).startswith("cuda")
        or str(name).startswith("void ")
        or str(name).startswith("VISUAL_ANCHOR_")
    ):
        continue

    events.append({
        "ts_us": float(ts),
        "source": "visual_chrome",
        "kind": ev.get("ph"),
        "name": str(name),
        "module": "",
        "dur_us": safe_float(ev.get("dur")),
        "details": json.dumps({
            "cat": ev.get("cat"),
            "pid": ev.get("pid"),
            "tid": ev.get("tid"),
            "args": ev.get("args", {}),
        }, ensure_ascii=False),
    })

# -------------------------------------------------
# Sort and write
# -------------------------------------------------
events.sort(key=lambda x: x["ts_us"])

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["ts_us", "source", "kind", "name", "module", "dur_us", "details"]
    )
    writer.writeheader()
    writer.writerows(events)

print(f"Wrote {len(events)} rows to {OUT_CSV}")
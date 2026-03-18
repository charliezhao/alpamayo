# stitch_parent_timeline.py
import csv
import json
from pathlib import Path

BASE = Path("/outputs")
PARENT_DIR = BASE / "parent_trace_20260306"

PARENT_FWD = PARENT_DIR / "parent_forward_debug.jsonl"
VIS_CALL = PARENT_DIR / "visual_call_debug.jsonl"
PARENT_CHROME = PARENT_DIR / "parent_trace_chrome.json"

OUT_CSV = BASE / "stitched_parent_timeline.csv"


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


# -------------------------------------------------
# Load jsonl logs
# -------------------------------------------------
parent_rows = load_jsonl(PARENT_FWD)
visual_call_rows = load_jsonl(VIS_CALL)

# -------------------------------------------------
# Find perf anchors from parent log
# -------------------------------------------------
perf_begin = None
perf_end = None
for r in parent_rows:
    if r.get("event") == "parent_forward_enter":
        perf_begin = r.get("ts_perf_ns")
    elif r.get("event") == "parent_forward_exit":
        perf_end = r.get("ts_perf_ns")

if perf_begin is None or perf_end is None:
    raise RuntimeError("Could not find parent perf anchors")

# -------------------------------------------------
# Load chrome trace and find profiler anchors
# -------------------------------------------------
with open(PARENT_CHROME, "r", encoding="utf-8") as f:
    chrome = json.load(f)

trace_events = chrome.get("traceEvents", [])

chrome_begin = None
chrome_end = None

for ev in trace_events:
    name = ev.get("name")
    ts = ev.get("ts")
    if name == "PARENT_ANCHOR_BEGIN" and ts is not None:
        chrome_begin = float(ts)
    elif name == "PARENT_ANCHOR_END" and ts is not None:
        chrome_end = float(ts)

if chrome_begin is None or chrome_end is None:
    raise RuntimeError("Could not find chrome parent anchors")

# -------------------------------------------------
# affine map: chrome_ts_us = a * perf_ns + b
# -------------------------------------------------
a = (chrome_end - chrome_begin) / (perf_end - perf_begin)
b = chrome_begin - a * perf_begin

print("Parent alignment solved:")
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
# 1. parent forward jsonl
# -------------------------------------------------
for r in parent_rows:
    ts_perf_ns = r.get("ts_perf_ns")
    if ts_perf_ns is None:
        continue

    dur_ns = r.get("dur_ns")
    dur_us = dur_ns / 1000.0 if dur_ns is not None else None

    events.append({
        "ts_us": perf_ns_to_chrome_us(ts_perf_ns),
        "source": "parent_forward",
        "kind": r.get("event"),
        "name": r.get("parent_path", "vlm.model.forward"),
        "module": r.get("parent_type", ""),
        "dur_us": dur_us,
        "details": json.dumps(r, ensure_ascii=False),
    })

# -------------------------------------------------
# 2. visual call jsonl
# -------------------------------------------------
for r in visual_call_rows:
    ts_perf_ns = r.get("ts_perf_ns")
    if ts_perf_ns is None:
        continue

    dur_ns = r.get("dur_ns")
    dur_us = dur_ns / 1000.0 if dur_ns is not None else None

    events.append({
        "ts_us": perf_ns_to_chrome_us(ts_perf_ns),
        "source": "visual_call",
        "kind": r.get("event"),
        "name": r.get("visual_path", "vlm.model.visual"),
        "module": "",
        "dur_us": dur_us,
        "details": json.dumps(r, ensure_ascii=False),
    })

# -------------------------------------------------
# 3. parent chrome trace
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
        or str(name).startswith("PARENT_")
        or str(name).startswith("VISUAL_CALL_")
    ):
        continue

    events.append({
        "ts_us": float(ts),
        "source": "parent_chrome",
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
# sort and write
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

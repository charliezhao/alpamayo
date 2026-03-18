#!/usr/bin/env python3
import json
import argparse
import sys
from typing import List


DEFAULT_DROP_PREFIX = [
    "aten::is_nonzero",
    "aten::item",
    "aten::_local_scalar_dense",
]

def should_drop_event(ev, drop_cpu_ops, drop_prefix, drop_cuda_sync):
    name = ev.get("name", "")
    cat = ev.get("cat", "")

    # Always keep your module annotations
    if isinstance(name, str) and name.startswith("MOD::"):
        return False

    # Drop all cpu_op if requested
    if drop_cpu_ops and cat == "cpu_op":
        return True

    # Drop specific aten ops
    if isinstance(name, str):
        for p in drop_prefix:
            if name.startswith(p):
                return True

    # Optional: drop cudaStreamSynchronize markers
    if drop_cuda_sync and isinstance(name, str) and name.startswith("cudaStreamSynchronize"):
        return True

    return False


def clean_trace(
    trace_in: str,
    trace_out: str,
    drop_cpu_ops: bool,
    drop_prefix: List[str],
    drop_cuda_sync: bool,
):
    print(f"[clean_trace] loading {trace_in}")

    with open(trace_in, "r") as f:
        obj = json.load(f)

    if isinstance(obj, dict) and "traceEvents" in obj:
        events = obj["traceEvents"]
        wrap_dict = True
    elif isinstance(obj, list):
        events = obj
        wrap_dict = False
    else:
        print("Unsupported chrome trace format.", file=sys.stderr)
        sys.exit(1)

    original_count = len(events)
    kept = []

    for ev in events:
        if not should_drop_event(ev, drop_cpu_ops, drop_prefix, drop_cuda_sync):
            kept.append(ev)

    print(f"[clean_trace] original events: {original_count}")
    print(f"[clean_trace] kept events    : {len(kept)}")
    print(f"[clean_trace] removed events : {original_count - len(kept)}")

    if wrap_dict:
        out_obj = {"traceEvents": kept}
    else:
        out_obj = kept

    with open(trace_out, "w") as f:
        json.dump(out_obj, f)

    print(f"[clean_trace] wrote cleaned trace to {trace_out}")


def main():
    parser = argparse.ArgumentParser(description="Clean PyTorch Chrome trace (remove noisy aten CPU ops).")
    parser.add_argument("trace_in", help="Input chrome trace JSON")
    parser.add_argument("trace_out", help="Output cleaned chrome trace JSON")

    parser.add_argument("--drop_cpu_ops", action="store_true",
                        help="Remove all cpu_op events (recommended for phase visualization)")

    parser.add_argument("--drop_cuda_sync", action="store_true",
                        help="Remove cudaStreamSynchronize events")

    parser.add_argument("--keep_is_nonzero", action="store_true",
                        help="Keep aten::is_nonzero (override default drop list)")

    args = parser.parse_args()

    drop_prefix = DEFAULT_DROP_PREFIX.copy()

    if args.keep_is_nonzero:
        drop_prefix = [p for p in drop_prefix if not p.startswith("aten::is_nonzero")]

    clean_trace(
        trace_in=args.trace_in,
        trace_out=args.trace_out,
        drop_cpu_ops=args.drop_cpu_ops,
        drop_prefix=drop_prefix,
        drop_cuda_sync=args.drop_cuda_sync,
    )


if __name__ == "__main__":
    main()
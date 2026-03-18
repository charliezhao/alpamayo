# parent_trace_helper.py
import csv
import json
import os
import time
from typing import Any, Dict, List

import torch


def _perf_ns() -> int:
    return time.perf_counter_ns()


def _is_tensor(x: Any) -> bool:
    return isinstance(x, torch.Tensor)


def _tensor_desc(x: torch.Tensor) -> Dict[str, Any]:
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype).replace("torch.", ""),
        "device": str(x.device),
        "bytes": int(x.numel() * x.element_size()),
        "requires_grad": bool(x.requires_grad),
    }


def _obj_desc(x: Any, depth: int = 0, max_depth: int = 2) -> Any:
    if depth > max_depth:
        return {"type": type(x).__name__, "truncated": True}

    if _is_tensor(x):
        return _tensor_desc(x)

    if isinstance(x, (list, tuple)):
        return {
            "type": type(x).__name__,
            "len": len(x),
            "items": [_obj_desc(v, depth + 1, max_depth) for v in x[:8]],
            "truncated": len(x) > 8,
        }

    if isinstance(x, dict):
        items = list(x.items())[:24]
        return {
            "type": "dict",
            "len": len(x),
            "items": {str(k): _obj_desc(v, depth + 1, max_depth) for k, v in items},
            "truncated": len(x) > 24,
        }

    if x is None:
        return None

    if isinstance(x, (int, float, bool, str)):
        return x

    return {"type": type(x).__name__, "repr": repr(x)[:200]}


def _resolve_parent_and_visual(model: torch.nn.Module):
    parent_candidates = [
        "vlm.model",   # expected for your Alpamayo case
        "model",
        "vlm",
    ]

    visual_candidates = [
        "vlm.model.visual",
        "model.visual",
        "visual",
    ]

    parent_module = None
    visual_module = None
    parent_path = None
    visual_path = None

    for path in parent_candidates:
        cur = model
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and isinstance(cur, torch.nn.Module):
            parent_module = cur
            parent_path = path
            break

    for path in visual_candidates:
        cur = model
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and isinstance(cur, torch.nn.Module):
            visual_module = cur
            visual_path = path
            break

    if parent_module is None:
        raise RuntimeError(f"Could not resolve parent module from {parent_candidates}")
    if visual_module is None:
        raise RuntimeError(f"Could not resolve visual module from {visual_candidates}")

    return parent_module, visual_module, parent_path, visual_path


class ParentTraceRecorder:
    def __init__(self, model: torch.nn.Module, out_dir: str, enable_once: bool = True):
        self.model = model
        self.out_dir = out_dir
        self.enable_once = enable_once
        self.used = False

        os.makedirs(self.out_dir, exist_ok=True)

        self.parent_module, self.visual_module, self.parent_path, self.visual_path = _resolve_parent_and_visual(model)

        self.parent_log_path = os.path.join(self.out_dir, "parent_forward_debug.jsonl")
        self.visual_call_log_path = os.path.join(self.out_dir, "visual_call_debug.jsonl")
        self.aten_csv_path = os.path.join(self.out_dir, "parent_aten_ops.csv")
        self.trace_json_path = os.path.join(self.out_dir, "parent_trace_chrome.json")

        self._orig_parent_forward = None
        self._orig_visual_forward = None

    def _write_jsonl(self, path: str, row: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _export_profiler_events(self, prof: torch.profiler.profile):
        try:
            prof.export_chrome_trace(self.trace_json_path)
        except Exception as e:
            self._write_jsonl(self.parent_log_path, {"event": "export_chrome_trace_error", "error": repr(e)})

        rows: List[Dict[str, Any]] = []
        try:
            events = prof.events()
        except Exception as e:
            self._write_jsonl(self.parent_log_path, {"event": "prof_events_error", "error": repr(e)})
            events = []

        for evt in events:
            name = getattr(evt, "name", "")
            if not name:
                continue

            if not (
                name.startswith("aten::")
                or name.startswith("cuda")
                or name.startswith("ampere")
                or name.startswith("void ")
                or name.startswith("PARENT_")
                or name.startswith("VISUAL_CALL_")
            ):
                continue

            rows.append({
                "name": name,
                "device_type": str(getattr(evt, "device_type", "")),
                "cpu_time_total_us": float(getattr(evt, "cpu_time_total", 0.0) or 0.0),
                "self_cpu_time_total_us": float(getattr(evt, "self_cpu_time_total", 0.0) or 0.0),
                "device_time_total_us": float(getattr(evt, "device_time_total", 0.0) or 0.0),
                "self_device_time_total_us": float(getattr(evt, "self_device_time_total", 0.0) or 0.0),
                "count": int(getattr(evt, "count", 1) or 1),
                "input_shapes": repr(getattr(evt, "input_shapes", None)),
                "stack": " | ".join(getattr(evt, "stack", [])[:8]) if getattr(evt, "stack", None) else "",
            })

        with open(self.aten_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "name",
                    "device_type",
                    "cpu_time_total_us",
                    "self_cpu_time_total_us",
                    "device_time_total_us",
                    "self_device_time_total_us",
                    "count",
                    "input_shapes",
                    "stack",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    def _wrap_visual_forward_for_parent(self):
        recorder = self
        self._orig_visual_forward = self.visual_module.forward

        def wrapped_visual_forward(*args, **kwargs):
            ts0 = _perf_ns()
            recorder._write_jsonl(
                recorder.visual_call_log_path,
                {
                    "event": "visual_call_enter",
                    "ts_perf_ns": ts0,
                    "visual_path": recorder.visual_path,
                    "args_desc": _obj_desc(args),
                    "kwargs_desc": _obj_desc(kwargs),
                },
            )

            with torch.autograd.profiler.record_function("VISUAL_CALL_BEGIN"):
                pass

            out = recorder._orig_visual_forward(*args, **kwargs)

            with torch.autograd.profiler.record_function("VISUAL_CALL_END"):
                pass

            ts1 = _perf_ns()
            recorder._write_jsonl(
                recorder.visual_call_log_path,
                {
                    "event": "visual_call_exit",
                    "ts_perf_ns": ts1,
                    "visual_path": recorder.visual_path,
                    "out_desc": _obj_desc(out),
                    "dur_ns": ts1 - ts0,
                },
            )
            return out

        self.visual_module.forward = wrapped_visual_forward

    def _unwrap_visual_forward(self):
        if self._orig_visual_forward is not None:
            self.visual_module.forward = self._orig_visual_forward
            self._orig_visual_forward = None

    def _wrap_parent_forward(self):
        recorder = self
        self._orig_parent_forward = self.parent_module.forward

        def wrapped_parent_forward(*args, **kwargs):
            if recorder.enable_once and recorder.used:
                return recorder._orig_parent_forward(*args, **kwargs)

            recorder.used = True

            ts_enter = _perf_ns()
            recorder._write_jsonl(
                recorder.parent_log_path,
                {
                    "event": "parent_forward_enter",
                    "ts_perf_ns": ts_enter,
                    "parent_path": recorder.parent_path,
                    "parent_type": type(recorder.parent_module).__name__,
                    "args_desc": _obj_desc(args),
                    "kwargs_desc": _obj_desc(kwargs),
                },
            )

            recorder._wrap_visual_forward_for_parent()

            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)

            try:
                with torch.profiler.profile(
                    activities=activities,
                    record_shapes=True,
                    with_stack=True,
                    profile_memory=True,
                ) as prof:
                    with torch.autograd.profiler.record_function("PARENT_ANCHOR_BEGIN"):
                        pass

                    out = recorder._orig_parent_forward(*args, **kwargs)

                    with torch.autograd.profiler.record_function("PARENT_ANCHOR_END"):
                        pass
            finally:
                recorder._unwrap_visual_forward()

            ts_exit = _perf_ns()
            recorder._write_jsonl(
                recorder.parent_log_path,
                {
                    "event": "parent_forward_exit",
                    "ts_perf_ns": ts_exit,
                    "parent_path": recorder.parent_path,
                    "out_desc": _obj_desc(out),
                    "dur_ns": ts_exit - ts_enter,
                },
            )

            recorder._export_profiler_events(prof)
            return out

        self.parent_module.forward = wrapped_parent_forward

    def install(self):
        self._wrap_parent_forward()
        self._write_jsonl(
            self.parent_log_path,
            {
                "event": "installed",
                "ts_perf_ns": _perf_ns(),
                "parent_path": self.parent_path,
                "visual_path": self.visual_path,
                "out_dir": self.out_dir,
            },
        )

    def remove(self):
        self._unwrap_visual_forward()
        if self._orig_parent_forward is not None:
            self.parent_module.forward = self._orig_parent_forward
            self._orig_parent_forward = None


def install_parent_trace(model: torch.nn.Module, out_dir: str, enable_once: bool = True) -> ParentTraceRecorder:
    rec = ParentTraceRecorder(model=model, out_dir=out_dir, enable_once=enable_once)
    rec.install()
    return rec

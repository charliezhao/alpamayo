# visual_trace_helper.py
import csv
import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

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
        items = list(x.items())[:16]
        return {
            "type": "dict",
            "len": len(x),
            "items": {str(k): _obj_desc(v, depth + 1, max_depth) for k, v in items},
            "truncated": len(x) > 16,
        }

    if x is None:
        return None

    if isinstance(x, (int, float, bool, str)):
        return x

    return {"type": type(x).__name__, "repr": repr(x)[:200]}


class VisualTraceRecorder:
    """
    Records:
      1) visual.forward enter/exit
      2) all child nn.Module calls under visual subtree
      3) all ATen ops inside visual.forward via local torch.profiler
    """

    def __init__(self, visual_module: torch.nn.Module, out_dir: str, enable_once: bool = True):
        self.visual_module = visual_module
        self.out_dir = out_dir
        self.enable_once = enable_once
        self.used = False

        os.makedirs(self.out_dir, exist_ok=True)

        self.forward_log_path = os.path.join(self.out_dir, "visual_forward_debug.jsonl")
        self.module_log_path = os.path.join(self.out_dir, "visual_layers.jsonl")
        self.aten_csv_path = os.path.join(self.out_dir, "visual_aten_ops.csv")
        self.trace_json_path = os.path.join(self.out_dir, "visual_trace_chrome.json")

        self._module_handles = []
        self._orig_forward = None
        self._module_call_idx = defaultdict(int)

        # Build stable module-name map
        self._name_of = {}
        for name, mod in self.visual_module.named_modules():
            fq_name = "vlm.model.visual" if name == "" else f"vlm.model.visual.{name}"
            self._name_of[id(mod)] = fq_name

    def _write_jsonl(self, path: str, row: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _module_pre_hook(self, mod: torch.nn.Module, args):
        mod_name = self._name_of.get(id(mod), mod.__class__.__name__)
        call_idx = self._module_call_idx[mod_name]
        self._module_call_idx[mod_name] += 1
        mod._visual_trace_call_idx = call_idx
        mod._visual_trace_t0 = _perf_ns()

        self._write_jsonl(
            self.module_log_path,
            {
                "event": "module_enter",
                "ts_perf_ns": _perf_ns(),
                "module": mod_name,
                "module_call_idx": call_idx,
                "type": type(mod).__name__,
                "in_desc": _obj_desc(args),
            },
        )

    def _module_post_hook(self, mod: torch.nn.Module, args, output):
        mod_name = self._name_of.get(id(mod), mod.__class__.__name__)
        call_idx = getattr(mod, "_visual_trace_call_idx", -1)
        t0 = getattr(mod, "_visual_trace_t0", None)
        dt_ms = (_perf_ns() - t0) / 1e6 if t0 is not None else None

        self._write_jsonl(
            self.module_log_path,
            {
                "event": "module_exit",
                "ts_perf_ns": _perf_ns(),
                "module": mod_name,
                "module_call_idx": call_idx,
                "type": type(mod).__name__,
                "out_desc": _obj_desc(output),
                "wall_ms": dt_ms,
            },
        )

    def _install_module_hooks(self):
        # Hook every module in visual subtree, including parent visual module itself.
        for _, mod in self.visual_module.named_modules():
            self._module_handles.append(mod.register_forward_pre_hook(self._module_pre_hook))
            self._module_handles.append(mod.register_forward_hook(self._module_post_hook))

    def _export_profiler_events(self, prof: torch.profiler.profile):
        # Export chrome trace for timeline browsing
        try:
            prof.export_chrome_trace(self.trace_json_path)
        except Exception as e:
            self._write_jsonl(
                self.forward_log_path,
                {
                    "event": "export_chrome_trace_error",
                    "error": repr(e),
                },
            )

        rows: List[Dict[str, Any]] = []
        try:
            events = prof.events()
        except Exception as e:
            self._write_jsonl(
                self.forward_log_path,
                {
                    "event": "prof_events_error",
                    "error": repr(e),
                },
            )
            events = []

        for evt in events:
            name = getattr(evt, "name", "")
            if not name:
                continue

            # Keep ATen ops and selected profiler entries
            if not (
                name.startswith("aten::")
                or name.startswith("cuda")
                or name.startswith("ampere")
                or name.startswith("void ")
            ):
                continue

            row = {
                "name": name,
                "device_type": str(getattr(evt, "device_type", "")),
                "cpu_time_total_us": float(getattr(evt, "cpu_time_total", 0.0) or 0.0),
                "self_cpu_time_total_us": float(getattr(evt, "self_cpu_time_total", 0.0) or 0.0),
                "device_time_total_us": float(getattr(evt, "device_time_total", 0.0) or 0.0),
                "self_device_time_total_us": float(getattr(evt, "self_device_time_total", 0.0) or 0.0),
                "count": int(getattr(evt, "count", 1) or 1),
                "input_shapes": repr(getattr(evt, "input_shapes", None)),
                "stack": " | ".join(getattr(evt, "stack", [])[:8]) if getattr(evt, "stack", None) else "",
            }
            rows.append(row)

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

    def _wrap_forward(self):
        self._orig_forward = self.visual_module.forward
        recorder = self

        def wrapped_forward(*args, **kwargs):
            if recorder.enable_once and recorder.used:
                return recorder._orig_forward(*args, **kwargs)

            recorder.used = True

            recorder._write_jsonl(
                recorder.forward_log_path,
                {
                    "event": "visual_forward_enter",
                    "ts_perf_ns": _perf_ns(),
                    "args_desc": _obj_desc(args),
                    "kwargs_desc": _obj_desc(kwargs),
                },
            )

            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)

            """
            with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
            ) as prof:
                out = recorder._orig_forward(*args, **kwargs)
            """

            begin_perf_ns = _perf_ns()
            recorder._write_jsonl(
                recorder.forward_log_path,
                {
                    "event": "visual_anchor_begin",
                    "ts_perf_ns": begin_perf_ns,
                },
            )

            with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
            ) as prof:
                with torch.autograd.profiler.record_function("VISUAL_ANCHOR_BEGIN"):
                    pass

                out = recorder._orig_forward(*args, **kwargs)

                with torch.autograd.profiler.record_function("VISUAL_ANCHOR_END"):
                    pass

            end_perf_ns = _perf_ns()
            recorder._write_jsonl(
                recorder.forward_log_path,
                {
                    "event": "visual_anchor_end",
                    "ts_perf_ns": end_perf_ns,
                },
            )

            recorder._write_jsonl(
                recorder.forward_log_path,
                {
                    "event": "visual_forward_exit",
                    "ts_perf_ns": _perf_ns(),
                    "out_desc": _obj_desc(out),
                },
            )

            recorder._export_profiler_events(prof)
            return out

        self.visual_module.forward = wrapped_forward

    def install(self):
        self._install_module_hooks()
        self._wrap_forward()
        self._write_jsonl(
            self.forward_log_path,
            {
                "event": "installed",
                "ts_perf_ns": _perf_ns(),
                "visual_module_type": type(self.visual_module).__name__,
                "out_dir": self.out_dir,
            },
        )

    def remove(self):
        for h in self._module_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._module_handles = []

        if self._orig_forward is not None:
            self.visual_module.forward = self._orig_forward
            self._orig_forward = None


def _resolve_visual_module(model: torch.nn.Module) -> torch.nn.Module:
    """
    Tries common paths used in Alpamayo/Qwen wrappers.
    """
    candidates = [
        "vlm.model.visual",
        "model.visual",
        "visual",
    ]

    for path in candidates:
        cur = model
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and isinstance(cur, torch.nn.Module):
            return cur

    raise RuntimeError(
        "Could not resolve visual module. Tried: " + ", ".join(candidates)
    )


def install_visual_trace(model: torch.nn.Module, out_dir: str, enable_once: bool = True) -> VisualTraceRecorder:
    visual_module = _resolve_visual_module(model)
    rec = VisualTraceRecorder(visual_module=visual_module, out_dir=out_dir, enable_once=enable_once)
    rec.install()
    return rec

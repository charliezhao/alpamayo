#!/usr/bin/env python3
"""
nsys_stage_bench.py

Stage-level GPU profiling for Alpamayo VLM side, aligned to NVIDIA table:
- Vision encoder (best effort)
- Prefill (prompt/KV-cache build)
- Reasoning decode (N tokens, default 40)

Uses NVTX ranges + cudaProfilerStart/Stop for clean nsys captures.

Run under nsys with:
  nsys profile ... --capture-range=cudaProfilerApi ... -o /outputs/nsys_xxx python -u nsys_stage_bench.py ...
"""

import os, sys, time, json, argparse
import torch

# repo src layout (repo root is /workspace/alpamayo)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper
import physical_ai_av


def nvtx_push(msg: str):
    torch.cuda.nvtx.range_push(msg)

def nvtx_pop():
    torch.cuda.nvtx.range_pop()


def find_vision_module(vlm_model: torch.nn.Module):
    """
    Best-effort: find a likely vision encoder submodule.
    Tries common attributes; else falls back to module name contains "vision/visual/vit".
    """
    for attr in ["vision_tower", "vision_encoder", "visual", "vision_model", "vision", "image_encoder"]:
        if hasattr(vlm_model, attr):
            m = getattr(vlm_model, attr)
            if isinstance(m, torch.nn.Module):
                return m, f"attr:{attr}"

    candidates = []
    for name, m in vlm_model.named_modules():
        lname = name.lower()
        if any(k in lname for k in ["vision", "visual", "image_encoder", "vit"]):
            if sum(p.numel() for p in m.parameters(recurse=False)) > 0:
                candidates.append((name, m))
    if candidates:
        candidates.sort(key=lambda x: sum(p.numel() for p in x[1].parameters()), reverse=True)
        name, m = candidates[0]
        return m, f"named:{name}"
    return None, "not_found"


@torch.no_grad()
def manual_prefill_and_decode(
    vlm,
    input_ids,
    attention_mask=None,
    max_new_tokens=40,
    top_p=0.98,
    temperature=0.6,
    pad_token_id=None,
    extra_kwargs=None,
):
    """
    Manual decode loop to separate:
    - PREFILL: one forward pass over prompt building past_key_values
    - REASONING: token-by-token decode for max_new_tokens
    """
    if extra_kwargs is None:
        extra_kwargs = {}

    device = input_ids.device
    bsz = input_ids.shape[0]

    # ---- PREFILL ----
    nvtx_push("prefill")
    torch.cuda.synchronize()
    t0 = time.time()

    out = vlm(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
        **extra_kwargs,
    )
    past = out.past_key_values

    torch.cuda.synchronize()
    t1 = time.time()
    nvtx_pop()

    # ---- DECODE (Reasoning) ----
    next_input_ids = input_ids[:, -1:].contiguous()
    generated = []

    nvtx_push(f"reasoning_decode_{max_new_tokens}toks")
    torch.cuda.synchronize()
    t2 = time.time()

    for _ in range(max_new_tokens):
        out = vlm(
            input_ids=next_input_ids,
            attention_mask=None,
            use_cache=True,
            past_key_values=past,
            return_dict=True,
            **extra_kwargs,
        )
        past = out.past_key_values
        logits = out.logits[:, -1, :]  # [B, vocab]

        if temperature is not None and temperature > 0:
            logits = logits / float(temperature)

        probs = torch.softmax(logits, dim=-1)

        if top_p is not None and top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cdf = torch.cumsum(sorted_probs, dim=-1)

            cutoff = cdf > top_p
            cutoff[:, 1:] = cutoff[:, :-1].clone()
            cutoff[:, 0] = False

            sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            sampled_sorted = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_idx.gather(-1, sampled_sorted)
        else:
            next_token = torch.multinomial(probs, num_samples=1)

        generated.append(next_token)
        next_input_ids = next_token

    torch.cuda.synchronize()
    t3 = time.time()
    nvtx_pop()

    if generated:
        generated = torch.cat(generated, dim=1)
    else:
        generated = torch.empty((bsz, 0), dtype=torch.long, device=device)

    return {
        "prefill_s": t1 - t0,
        "reasoning_s": t3 - t2,
        "generated_ids": generated,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_id", required=True)
    ap.add_argument("--t0_us", type=int, default=5_100_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--maybe_stream", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--top_p", type=float, default=0.98)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--profile_iters", type=int, default=1)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required for nsys profiling."

    amp_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    # Dataset interface (your load_physical_aiavdataset already supports PHYSICALAI_LOCAL_DIR)
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    # Model
    model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=amp_dtype).to(args.device)
    model.eval()
    processor = helper.get_processor(model.tokenizer)

    # Prepare one sample
    data = load_physical_aiavdataset(
        args.clip_id, t0_us=args.t0_us, avdi=avdi, maybe_stream=args.maybe_stream
    )
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    tokenized = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    tokenized = helper.to_device(tokenized, args.device)

    if "input_ids" not in tokenized:
        raise KeyError(
            f"tokenized missing input_ids. Keys={list(tokenized.keys())}. "
            "Do NOT convert the BatchEncoding to a plain dict incorrectly."
        )

    input_ids = tokenized["input_ids"]
    attn_mask = tokenized.get("attention_mask", None)
    extra_kwargs = {k: v for k, v in tokenized.items() if k != "input_ids"}

    # Warmup (not profiled)
    for _ in range(args.warmup):
        with torch.autocast(args.device, dtype=amp_dtype):
            _ = model.vlm(
                input_ids=input_ids,
                attention_mask=attn_mask,
                use_cache=True,
                return_dict=True,
                **extra_kwargs,
            )
        torch.cuda.synchronize()

    # Stage 1: Vision encoder (best effort)
    vlm_core = model.vlm.model if hasattr(model.vlm, "model") else model.vlm
    vision_mod, vision_tag = find_vision_module(vlm_core)

    vision_time_s = None
    if vision_mod is not None:
        pixel_values = extra_kwargs.get("pixel_values", None) or extra_kwargs.get("images", None)
        if pixel_values is not None:
            nvtx_push(f"vision_encoder({vision_tag})")
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.autocast(args.device, dtype=amp_dtype):
                _ = vision_mod(pixel_values)
            torch.cuda.synchronize()
            t1 = time.time()
            nvtx_pop()
            vision_time_s = t1 - t0

    # Profiled section (prefill + reasoning)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()

    results = []
    for _ in range(args.profile_iters):
        with torch.autocast(args.device, dtype=amp_dtype):
            r = manual_prefill_and_decode(
                model.vlm,
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=args.max_new_tokens,
                top_p=args.top_p,
                temperature=args.temperature,
                pad_token_id=model.tokenizer.pad_token_id,
                extra_kwargs=extra_kwargs,
            )
        results.append(r)

    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    prefill_s = sum(r["prefill_s"] for r in results) / len(results)
    reasoning_s = sum(r["reasoning_s"] for r in results) / len(results)

    print(json.dumps({
        "clip_id": args.clip_id,
        "dtype": args.dtype,
        "vision_encoder_s": vision_time_s,
        "prefill_s": prefill_s,
        "reasoning_s": reasoning_s,
        "reasoning_tokens": args.max_new_tokens,
        "note": "Use nsys NVTX ranges for GPU-kernel attribution. vision_encoder_s may be None if not isolatable.",
    }, indent=2))


if __name__ == "__main__":
    main()


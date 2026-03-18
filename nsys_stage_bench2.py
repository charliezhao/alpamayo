#!/usr/bin/env python3
"""
nsys_stage_bench2.py

Stage-level GPU profiling for Alpamayo VLM side (Thor container friendly).
Fixes kwarg collisions (attention_mask duplicated) seen with Qwen3-VL.

Captures NVTX ranges + cudaProfilerStart/Stop for clean nsys runs.

Stages we isolate (best effort):
- PREFILL: single forward pass over full prompt building KV cache
- REASONING: token-by-token decode for N tokens (default 40)

Vision encoder isolation is model-dependent; for Qwen3-VL it’s often fused into forward.
So we primarily rely on NVTX ranges and (optionally) later do “vision-only” ablations.
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


@torch.no_grad()
def manual_prefill_and_decode(
    vlm,
    model_kwargs: dict,
    max_new_tokens=40,
    top_p=0.98,
    temperature=0.6,
):
    """
    Manual decode loop to separate:
    - PREFILL: one forward pass over prompt building past_key_values
    - REASONING: token-by-token decode for max_new_tokens
    """
    # ----- PREFILL -----
    nvtx_push("prefill")
    torch.cuda.synchronize()
    t0 = time.time()

    out = vlm(use_cache=True, return_dict=True, **model_kwargs)
    past = out.past_key_values

    torch.cuda.synchronize()
    t1 = time.time()
    nvtx_pop()

    # We decode from the last token of the prompt
    input_ids = model_kwargs["input_ids"]
    next_input_ids = input_ids[:, -1:].contiguous()

    # Some VL models require extra vision-related kwargs only on prefill.
    # For decode we keep ONLY text-relevant kwargs.
    decode_kwargs = dict(model_kwargs)
    decode_kwargs["input_ids"] = next_input_ids
    decode_kwargs.pop("pixel_values", None)
    decode_kwargs.pop("images", None)
    decode_kwargs.pop("image_grid_thw", None)
    decode_kwargs.pop("pixel_values_videos", None)
    decode_kwargs.pop("video_grid_thw", None)

    nvtx_push(f"reasoning_decode_{max_new_tokens}toks")
    torch.cuda.synchronize()
    t2 = time.time()

    for _ in range(max_new_tokens):
        out = vlm(
            use_cache=True,
            return_dict=True,
            past_key_values=past,
            **decode_kwargs,
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

        decode_kwargs["input_ids"] = next_token

    torch.cuda.synchronize()
    t3 = time.time()
    nvtx_pop()

    return {
        "prefill_s": t1 - t0,
        "reasoning_s": t3 - t2,
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

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=amp_dtype).to(args.device)
    model.eval()
    processor = helper.get_processor(model.tokenizer)

    # Load one sample
    data = load_physical_aiavdataset(
        args.clip_id, t0_us=args.t0_us, avdi=avdi, maybe_stream=args.maybe_stream
    )

    # Build chat + tokenize
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

    # Build clean forward kwargs (NO duplicates)
    if "input_ids" not in tokenized:
        raise KeyError(f"tokenized missing input_ids. Keys={list(tokenized.keys())}")

    model_kwargs = dict(tokenized)  # BatchEncoding -> plain dict
    # Ensure no None values that some forwards dislike
    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

    # Warmup (not profiled)
    for _ in range(args.warmup):
        with torch.autocast(args.device, dtype=amp_dtype):
            _ = model.vlm(use_cache=True, return_dict=True, **model_kwargs)
        torch.cuda.synchronize()

    # Start capture
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()

    results = []
    for _ in range(args.profile_iters):
        with torch.autocast(args.device, dtype=amp_dtype):
            r = manual_prefill_and_decode(
                model.vlm,
                model_kwargs=model_kwargs,
                max_new_tokens=args.max_new_tokens,
                top_p=args.top_p,
                temperature=args.temperature,
            )
        results.append(r)

    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    prefill_s = sum(r["prefill_s"] for r in results) / len(results)
    reasoning_s = sum(r["reasoning_s"] for r in results) / len(results)

    print(json.dumps({
        "clip_id": args.clip_id,
        "dtype": args.dtype,
        "prefill_s": prefill_s,
        "reasoning_s": reasoning_s,
        "reasoning_tokens": args.max_new_tokens,
        "note": "Use nsys NVTX ranges to attribute GPU time. Vision encoder may be fused into prefill for Qwen3-VL.",
    }, indent=2))


if __name__ == "__main__":
    main()


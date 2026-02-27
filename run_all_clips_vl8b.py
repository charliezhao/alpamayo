#!/usr/bin/env python3
import argparse, json, os, time, glob
from pathlib import Path

import cv2
from PIL import Image

from transformers import AutoProcessor
from vllm import LLM, SamplingParams


def find_camera_mp4(clips_dir: str, clip_id: str, camera: str) -> str:
    # clip folders are like clips/clip_48ec6a11 (prefix = first 8 chars)
    clip_prefix = clip_id.split("-")[0]  # '48ec6a11'
    base = Path(clips_dir) / f"clip_{clip_prefix}" / "extracted" / "camera"

    if camera == "auto":
        preferred = [
            "camera_front_wide_120fov",
            "camera_front_120fov",
            "camera_cross_right_120fov",
            "camera_cross_left_120fov",
        ]
        for cam in preferred:
            cand = glob.glob(str(base / cam / f"{clip_id}.{cam}.mp4"))
            if cand:
                return cand[0]
        any_mp4 = glob.glob(str(base / "*" / f"{clip_id}.*.mp4"))
        if any_mp4:
            return any_mp4[0]
        raise RuntimeError(f"No camera mp4 found under {base} for clip_id={clip_id}")

    mp4 = base / camera / f"{clip_id}.{camera}.mp4"
    if not mp4.exists():
        raise RuntimeError(f"Missing mp4: {mp4}")
    return str(mp4)


def read_clip_ids(path: str, max_clips: int):
    ids = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(s)
            if max_clips and len(ids) >= max_clips:
                break
    return ids


def pick_frame_from_video(mp4_path: str, mode: str) -> Image.Image:
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open: {mp4_path}")

    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if n <= 0:
        ok, frame_bgr = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"No frames in: {mp4_path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    if mode == "first":
        idx = 0
    elif mode == "last":
        idx = max(n - 1, 0)
    else:
        idx = n // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {idx}/{n} from: {mp4_path}")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def build_qwen_vl_prompt(processor: AutoProcessor, user_text: str) -> str:
    """
    IMPORTANT:
    - We must create a chat-template prompt that includes the model's image placeholder tokens.
    - vLLM will replace those placeholders with the actual image(s) from multi_modal_data.
    """
    messages = [
        {
            "role": "user",
            "content": [
                # The actual string here doesn't matter much; it just marks an image slot.
                {"type": "image", "image": "placeholder"},
                {"type": "text", "text": user_text},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="e.g. Qwen/Qwen3-VL-8B-Instruct OR local nvfp4 model dir")
    ap.add_argument("--processor", default=None,
                    help="HF id/dir for AutoProcessor. Default: --model")
    ap.add_argument("--clip_ids_file", required=True)
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--out_jsonl", required=True)

    ap.add_argument("--max_clips", type=int, default=50)
    ap.add_argument("--frame_mode", choices=["first", "middle", "last"], default="middle")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--camera", default="auto",
                    help="camera folder name like camera_front_wide_120fov, or 'auto'")

    # vLLM memory knobs
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--max_num_seqs", type=int, default=1)
    ap.add_argument("--gpu_mem_util", type=float, default=0.90)
    ap.add_argument("--enforce_eager", action="store_true")

    # generation knobs
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)

    # optional prompt text override
    ap.add_argument("--question", default="What are the key objects and motions in this driving scene?")

    args = ap.parse_args()

    # Your preference: HF cache in ~/hf_cache (not ~/.cache)
    os.environ.setdefault("HF_HOME", str(Path.home() / "hf_cache"))
    os.environ.setdefault("HF_HUB_CACHE", str(Path.home() / "hf_cache" / "hub"))

    proc_id = args.processor or args.model
    processor = AutoProcessor.from_pretrained(proc_id)

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=args.enforce_eager,
        # IMPORTANT: tell vLLM we will send 1 image per prompt
        limit_mm_per_prompt={"image": {"count": 1, "width": 640, "height": 360}, "video": 0},
    )

    samp = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    clip_ids = read_clip_ids(args.clip_ids_file, args.max_clips)

    # Warmup (compile/alloc/cudagraph) on first clip
    if args.warmup > 0 and clip_ids:
        cid0 = clip_ids[0]
        mp4 = find_camera_mp4(args.clips_dir, cid0, args.camera)
        print("DEBUG warmup mp4 =", mp4)
        image0 = pick_frame_from_video(mp4, args.frame_mode)
        prompt0 = build_qwen_vl_prompt(processor, "Describe the scene briefly.")

        for _ in range(args.warmup):
            _ = llm.generate(
                {
                    "prompt": prompt0,
                    "multi_modal_data": {"image": [image0]},
                },
                sampling_params=samp,
            )

    outp = Path(args.out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)

    with outp.open("w") as f:
        for cid in clip_ids:
            t0 = time.perf_counter()
            ok = True
            err = None
            out_text = None

            try:
                mp4 = find_camera_mp4(args.clips_dir, cid, args.camera)
                image = pick_frame_from_video(mp4, args.frame_mode)
                prompt = build_qwen_vl_prompt(processor, args.question)

                outputs = llm.generate(
                    {
                        "prompt": prompt,
                        "multi_modal_data": {"image": [image]},
                    },
                    sampling_params=samp,
                )

                # outputs is a list; take first
                o0 = outputs[0]
                out_text = o0.outputs[0].text if o0.outputs else None

            except Exception as e:
                ok = False
                err = repr(e)

            t1 = time.perf_counter()

            rec = {
                "clip_id": cid,
                "ok": ok,
                "camera": args.camera,
                "frame_mode": args.frame_mode,
                "e2e_ms": (t1 - t0) * 1000.0,
                "model": args.model,
                "processor": proc_id,
                "max_model_len": args.max_model_len,
                "max_num_seqs": args.max_num_seqs,
                "gpu_mem_util": args.gpu_mem_util,
                "enforce_eager": args.enforce_eager,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "question": args.question,
                "text": out_text,
                "error": err,
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Download data for ONE clip UUID from NVIDIA PhysicalAI-Autonomous-Vehicles
without triggering 429.

- Uses clip_index.parquet index (clip_id is the DF index) -> split + chunk
- Uses metadata/sensor_presence.parquet to know which sensors exist for that row
- For each present sensor channel:
    * list only the relevant directory (recursive=False)
    * try to find files whose name contains clip_id (best case: per-clip assets)
    * else try to locate chunk subdir then pick a shard file (parquet/tar) (fallback)

This avoids any full recursive repo listing.
"""

import os
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

DATASET_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
REPO_TYPE = "dataset"

# We’ll accept these as "shard-ish" files if no per-clip filenames exist
SHARD_EXTS = (".parquet", ".tar", ".tar.gz", ".tgz")


def list_dir(api: HfApi, path_in_repo: str) -> List[str]:
    """Non-recursive listing of a single directory. Returns child paths or []."""
    try:
        items = api.list_repo_tree(
            repo_id=DATASET_REPO,
            repo_type=REPO_TYPE,
            path_in_repo=path_in_repo,
            recursive=False,
            expand=False,
        )
        return [it.path for it in items if getattr(it, "path", None)]
    except Exception:
        return []


def download_file(repo_path: str, out_dir: Path, token: Optional[str]) -> Optional[Path]:
    """Download one repo file to out_dir; return local path or None."""
    try:
        local = hf_hub_download(
            repo_id=DATASET_REPO,
            repo_type=REPO_TYPE,
            filename=repo_path,
            token=token,
            local_dir=str(out_dir),
            local_dir_use_symlinks=False,
        )
        return Path(local)
    except Exception:
        return None


def find_split_dir(api: HfApi, base: str, split: str) -> Optional[str]:
    """
    Given base like 'camera/camera_front_wide_120fov', find child directory
    that corresponds to split (train/val/test). Uses small listing.
    """
    children = list_dir(api, base)
    # Prefer exact match base/split
    want = f"{base}/{split}"
    if want in children:
        return want
    # Else pick any child that ends with '/split'
    for c in children:
        if c.endswith(f"/{split}"):
            return c
    return None


def find_chunk_dir(api: HfApi, split_dir: str, chunk: int) -> Optional[str]:
    """
    Try common chunk directory naming under split_dir.
    We do NOT recurse; we just list split_dir once.
    """
    children = list_dir(api, split_dir)
    cands = [
        f"{split_dir}/chunk={chunk}",
        f"{split_dir}/chunk_{chunk}",
        f"{split_dir}/{chunk}",
    ]
    for d in cands:
        if d in children:
            return d
    # Also match any child containing chunk token (best-effort)
    token = str(chunk)
    for c in children:
        if ("chunk=" + token) in c or ("chunk_" + token) in c or c.endswith("/" + token):
            return c
    return None


def choose_files_for_clip(api: HfApi, search_dir: str, clip_id: str) -> List[str]:
    """
    List one directory and return all files whose path contains clip_id.
    """
    items = list_dir(api, search_dir)
    return [p for p in items if clip_id in p]


def choose_one_shard(api: HfApi, search_dir: str) -> Optional[str]:
    """
    List one directory and pick one "shard-ish" file (parquet/tar) if present.
    Prefer .parquet first (often easiest to handle), otherwise tar.
    """
    items = list_dir(api, search_dir)
    files = [p for p in items if p.endswith(SHARD_EXTS)]
    if not files:
        return None
    # Prefer parquet
    files.sort(key=lambda p: (0 if p.endswith(".parquet") else 1, p))
    return files[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--clip_id", required=True, help="UUID clip id (must exist as index in clip_index.parquet)")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load indices locally
    clip_index = pd.read_parquet(dataset_dir / "clip_index.parquet")
    if args.clip_id not in clip_index.index:
        raise SystemExit(f"clip_id not found in clip_index.parquet index: {args.clip_id}")

    row = clip_index.loc[args.clip_id]
    split = str(row["split"])
    chunk = int(row["chunk"])
    print(f"clip_id={args.clip_id}")
    print(f"split={split}, chunk={chunk}")

    # Sensor presence: row-aligned with clip_index
    # If the index matches, we can align by index; otherwise fallback to positional alignment
    sp_path = dataset_dir / "metadata" / "sensor_presence.parquet"
    sp = pd.read_parquet(sp_path)

    if args.clip_id in sp.index:
        sp_row = sp.loc[args.clip_id]
    else:
        # positional alignment fallback
        # (works if both were built with same row order)
        try:
            pos = clip_index.index.get_loc(args.clip_id)
            sp_row = sp.iloc[pos]
        except Exception:
            sp_row = None

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    # Build list of sensor channels to try
    channels = []

    # Cameras
    cam_root = "camera"
    cam_names = [
        "camera_cross_left_120fov",
        "camera_cross_right_120fov",
        "camera_front_tele_30fov",
        "camera_front_wide_120fov",
        "camera_rear_left_70fov",
        "camera_rear_right_70fov",
        "camera_rear_tele_30fov",
    ]
    for cam in cam_names:
        if sp_row is None or bool(sp_row.get(cam, True)):
            channels.append((f"{cam_root}/{cam}", "camera", cam))

    # LiDAR
    if sp_row is None or bool(sp_row.get("lidar_top_360fov", True)):
        channels.append(("lidar/lidar_top_360fov", "lidar", "lidar_top_360fov"))

    # Radar (a few; you can add more if needed)
    radar_root = "radar"
    for col in sp.columns:
        if col.startswith("radar_") and col != "radar_config":
            if sp_row is None or bool(sp_row.get(col, True)):
                channels.append((f"{radar_root}/{col}", "radar", col))

    print(f"Will probe {len(channels)} sensor-channel folders (bounded, non-recursive).")

    # For each channel folder:
    # 1) find split dir
    # 2) find chunk dir (if exists)
    # 3) try to download files matching clip_id (best case)
    # 4) else download one shard file as fallback
    for base, modality, name in channels:
        print(f"\n== {modality}:{name} ==")
        split_dir = find_split_dir(api, base, split)
        if not split_dir:
            print(f"  no split dir found under {base} for {split}")
            continue

        # Prefer chunk dir if present; otherwise search directly in split dir
        chunk_dir = find_chunk_dir(api, split_dir, chunk)
        search_dir = chunk_dir if chunk_dir else split_dir

        if chunk_dir:
            print(f"  using chunk dir: {chunk_dir}")
        else:
            print(f"  no chunk dir found; using split dir: {split_dir}")

        # Try per-clip files
        matches = choose_files_for_clip(api, search_dir, args.clip_id)
        if matches:
            print(f"  found {len(matches)} file(s) containing clip_id; downloading...")
            for rp in matches:
                local = download_file(rp, out_dir / "downloaded" / modality / name, token)
                print(f"   - {rp} -> {local}")
            continue

        # Fallback: download one shard file from that directory
        shard = choose_one_shard(api, search_dir)
        if shard:
            print(f"  no per-clip filenames; downloading one shard as fallback: {shard}")
            local = download_file(shard, out_dir / "shards" / modality / name, token)
            print(f"   - {shard} -> {local}")
        else:
            print("  no per-clip files and no shard files found in this directory")

    print("\nDONE. Check:")
    print(f"  {out_dir}/downloaded/   (if per-clip assets exist)")
    print(f"  {out_dir}/shards/       (fallback shards if per-clip assets don’t exist)")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import zipfile
import pandas as pd
from huggingface_hub import hf_hub_download

DATASET_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
REPO_TYPE = "dataset"

CAMERAS = [
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
    "camera_front_wide_120fov",
    "camera_rear_left_70fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
]

def download(repo_path: str, out_dir: Path, token: str | None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    local = hf_hub_download(
        repo_id=DATASET_REPO,
        repo_type=REPO_TYPE,
        filename=repo_path,
        token=token,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
    )
    return Path(local)

def extract_matching(zip_path: Path, clip_id: str, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    hits = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        # Save listing for debugging
        (out_dir / "_zip_listing.txt").write_text("\n".join(names))

        for n in names:
            if clip_id in n:
                zf.extract(n, path=out_dir)
                hits += 1
    return hits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--clip_id", required=True)
    ap.add_argument("--limit_radar", type=int, default=6, help="download at most N radar channels (to keep downloads small)")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_index = pd.read_parquet(dataset_dir / "clip_index.parquet")
    if args.clip_id not in clip_index.index:
        raise SystemExit(f"clip_id not found in clip_index.parquet index: {args.clip_id}")

    row = clip_index.loc[args.clip_id]
    chunk = int(row["chunk"])
    split = str(row["split"])
    chunk4 = f"{chunk:04d}"

    print(f"clip_id={args.clip_id}")
    print(f"split={split} (metadata only), chunk={chunk} -> shard suffix chunk_{chunk4}.zip")

    sp = pd.read_parquet(dataset_dir / "metadata" / "sensor_presence.parquet")
    # sensor_presence might not have UUID index; try align by position if needed
    if args.clip_id in sp.index:
        sp_row = sp.loc[args.clip_id]
    else:
        pos = clip_index.index.get_loc(args.clip_id)
        sp_row = sp.iloc[pos]

    token = os.environ.get("HF_TOKEN")

    # Build list of sensor zip paths to download
    jobs = []

    # Cameras
    for cam in CAMERAS:
        if bool(sp_row.get(cam, True)):
            repo_path = f"camera/{cam}/{cam}.chunk_{chunk4}.zip"
            jobs.append(("camera", cam, repo_path))

    # LiDAR
    if bool(sp_row.get("lidar_top_360fov", True)):
        lid = "lidar_top_360fov"
        repo_path = f"lidar/{lid}/{lid}.chunk_{chunk4}.zip"
        jobs.append(("lidar", lid, repo_path))

    # Radar: many channels; keep bounded
    radar_cols = [c for c in sp.index if False]  # unused
    radar_channels = [c for c in sp.columns if c.startswith("radar_") and c != "radar_config"]
    selected_radars = []
    for r in radar_channels:
        if bool(sp_row.get(r, True)):
            selected_radars.append(r)
    selected_radars = selected_radars[: max(0, args.limit_radar)]

    for r in selected_radars:
        repo_path = f"radar/{r}/{r}.chunk_{chunk4}.zip"
        jobs.append(("radar", r, repo_path))

    print(f"Planned downloads: {len(jobs)} zip shard(s) (radar limited to {len(selected_radars)})")

    # Download + extract
    for modality, name, repo_path in jobs:
        print(f"\n== {modality}:{name} ==")
        try:
            local = download(repo_path, out_dir / "shards" / modality / name, token)
        except Exception as e:
            print(f"  download failed: {repo_path}\n  error: {e}")
            continue

        print(f"  downloaded: {repo_path}")
        hits = extract_matching(local, args.clip_id, out_dir / "extracted" / modality / name)
        if hits > 0:
            print(f"  extracted {hits} file(s) matching clip_id")
        else:
            print(f"  extracted 0 files matching clip_id")
            print(f"  saved zip listing to: {out_dir / 'extracted' / modality / name / '_zip_listing.txt'}")

    print("\nDONE.")
    print("Check extracted files under:", out_dir / "extracted")

if __name__ == "__main__":
    main()


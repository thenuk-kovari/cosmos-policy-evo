#!/usr/bin/env python3
"""Build and verify random-access JPEG sidecars for converted datasets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cosmos_policy.datasets.indexed_frame_store import CAMERAS, build_from_videos, load_frames


def _read_path(dataset) -> str:
    value = dataset[()]
    return value.decode() if isinstance(value, bytes) else str(value)


def episode_inputs(hdf5_path: Path) -> tuple[dict[str, str], int]:
    with h5py.File(hdf5_path, "r") as handle:
        paths = handle["observations/video_paths"]
        video_paths = {camera: str(hdf5_path.parent / _read_path(paths[camera])) for camera in CAMERAS}
        frames = int(handle["action"].shape[0])
        if handle["observations/qpos"].shape[0] != frames:
            raise ValueError(f"{hdf5_path}: action/proprio length mismatch")
    return video_paths, frames


def convert_episode(hdf5_path: Path, quality: int, overwrite: bool) -> dict:
    video_paths, frames = episode_inputs(hdf5_path)
    store = build_from_videos(
        hdf5_path, video_paths, frames, jpeg_quality=quality, overwrite=overwrite
    )
    return {
        "episode": str(hdf5_path),
        "frames": frames,
        "bytes": Path(store["data_path"]).stat().st_size,
        "sha256": store["sha256"],
    }


def psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    mse = float(np.mean((reference.astype(np.float32) - candidate.astype(np.float32)) ** 2))
    return float("inf") if mse == 0 else float(20 * np.log10(255.0) - 10 * np.log10(mse))


def load_reference_frames(video_path: str, frame_indices: list[int]) -> np.ndarray:
    """Sequentially decode selected source frames for integrity verification."""
    requested = set(frame_indices)
    found: dict[int, np.ndarray] = {}
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"could not open verification video {video_path}")
    try:
        frame_index = 0
        while requested - found.keys():
            ok, bgr = capture.read()
            if not ok:
                break
            if frame_index in requested:
                found[frame_index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frame_index += 1
    finally:
        capture.release()
    missing = sorted(requested - found.keys())
    if missing:
        raise IndexError(f"{video_path}: verification frames absent: {missing}")
    return np.stack([found[index] for index in frame_indices])


def verify_episode(hdf5_path: Path, sample_count: int, seed: int, minimum_psnr_db: float) -> dict:
    video_paths, frames = episode_inputs(hdf5_path)
    from cosmos_policy.datasets.indexed_frame_store import load_index

    store = load_index(hdf5_path, expected_frames=frames)
    if store is None:
        raise FileNotFoundError(f"{hdf5_path}: frame-store index absent")
    rng = np.random.default_rng(seed)
    indices = np.unique(rng.integers(0, frames, size=min(sample_count, frames))).tolist()
    camera_metrics = {}
    for camera in CAMERAS:
        reference = load_reference_frames(video_paths[camera], indices)
        candidate = load_frames(store, camera, indices)
        if reference.shape != candidate.shape:
            raise ValueError(f"{hdf5_path}: {camera} shape mismatch")
        difference = np.abs(reference.astype(np.int16) - candidate.astype(np.int16))
        camera_psnr = psnr(reference, candidate)
        if camera_psnr < minimum_psnr_db:
            raise ValueError(
                f"{hdf5_path}: {camera} sidecar PSNR {camera_psnr:.2f} dB "
                f"is below {minimum_psnr_db:.2f} dB"
            )
        camera_metrics[camera] = {
            "frames": len(indices),
            "mae": float(difference.mean()),
            "max_abs": int(difference.max(initial=0)),
            "psnr_db": camera_psnr,
        }
    return {"episode": str(hdf5_path), "cameras": camera_metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="Converted dataset root containing train/ and val/")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-episodes", type=int, default=3)
    parser.add_argument("--verify-frames", type=int, default=12)
    parser.add_argument("--minimum-psnr-db", type=float, default=35.0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    episodes = sorted(args.dataset.glob("train/*.hdf5")) + sorted(args.dataset.glob("val/*.hdf5"))
    if not episodes:
        raise FileNotFoundError(f"no HDF5 episodes under {args.dataset}")
    converted = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(convert_episode, episode, args.jpeg_quality, args.overwrite): episode
            for episode in episodes
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            converted.append(result)
            print(f"[{completed}/{len(episodes)}] {result['episode']} ({result['bytes'] / 2**30:.2f} GiB)", flush=True)

    verified = [
        verify_episode(episode, args.verify_frames, seed=index, minimum_psnr_db=args.minimum_psnr_db)
        for index, episode in enumerate(episodes[: args.verify_episodes])
    ]
    report = {
        "format": "indexed_jpeg_sidecar_v1",
        "dataset": str(args.dataset.resolve()),
        "episodes": len(episodes),
        "frames": sum(item["frames"] for item in converted),
        "bytes": sum(item["bytes"] for item in converted),
        "jpeg_quality": args.jpeg_quality,
        "minimum_psnr_db": args.minimum_psnr_db,
        "converted": sorted(converted, key=lambda item: item["episode"]),
        "verification": verified,
    }
    report_path = args.report or args.dataset / "indexed_frame_store_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"converted", "verification"}}, indent=2))
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()

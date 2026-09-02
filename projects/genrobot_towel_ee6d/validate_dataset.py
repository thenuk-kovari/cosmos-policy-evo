#!/usr/bin/env python3
"""Validate converted GenRobot data before spending GPU time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

from cosmos_policy.datasets.ee_q0_actions import (
    CONTRACT_NAME,
    NORMALIZATION_CONTRACT,
    SHARED_ACTION_DIM,
    STORAGE_CONTRACT,
    UMI_ACTIVE_DIMS,
    canonical_shared_statistics,
    shared_q0_action_extrema,
    validate_shared_pose_source,
)


def video_frames(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open {path}")
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--minimum-episodes", type=int, default=1)
    args = parser.parse_args()
    expected_statistics = {key: value.tolist() for key, value in canonical_shared_statistics().items()}
    if json.loads((args.data_dir / "dataset_statistics.json").read_text()) != expected_statistics:
        raise ValueError("dataset_statistics.json is not the canonical fixed contract")
    files = sorted(args.data_dir.glob("*/*.hdf5"))
    if len(files) < args.minimum_episodes:
        raise ValueError(f"found only {len(files)} episodes; require {args.minimum_episodes}")
    expected_mask = np.zeros(SHARED_ACTION_DIM, dtype=np.float32)
    expected_mask[UMI_ACTIVE_DIMS] = 1.0
    expected_proprio_mask = np.zeros(17, dtype=np.float32)
    expected_proprio_mask[[7, 15]] = 1.0
    split_counts: dict[str, int] = {}
    total_frames = 0
    extrema_min = np.full(SHARED_ACTION_DIM, np.inf)
    extrema_max = np.full(SHARED_ACTION_DIM, -np.inf)
    for path in files:
        split_counts[path.parent.name] = split_counts.get(path.parent.name, 0) + 1
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("action_contract") != CONTRACT_NAME:
                raise ValueError(f"{path}: action contract mismatch")
            if handle.attrs.get("action_storage_contract") != STORAGE_CONTRACT:
                raise ValueError(f"{path}: storage contract mismatch")
            if handle.attrs.get("normalization_contract") != NORMALIZATION_CONTRACT:
                raise ValueError(f"{path}: normalization contract mismatch")
            source = validate_shared_pose_source(handle["action"][:])
            proprio = handle["observations/qpos"][:]
            mask = handle["action_dim_mask"][:]
            if proprio.shape != (len(source), 17) or not np.isfinite(proprio).all():
                raise ValueError(f"{path}: invalid 17-D proprio")
            if not np.array_equal(mask, expected_mask):
                raise ValueError(f"{path}: wrong action availability mask")
            if not np.array_equal(handle["proprio_dim_mask"][:], expected_proprio_mask):
                raise ValueError(f"{path}: wrong proprio availability mask")
            for relative in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                value = handle[f"observations/video_paths/{relative}"][()]
                filename = value.decode() if isinstance(value, bytes) else str(value)
                count = video_frames(path.parent / filename)
                if count != len(source):
                    raise ValueError(f"{path}: {relative} has {count} frames, expected {len(source)}")
            episode_min, episode_max = shared_q0_action_extrema(source)
            extrema_min = np.minimum(extrema_min, episode_min)
            extrema_max = np.maximum(extrema_max, episode_max)
            total_frames += len(source)
    statistics = canonical_shared_statistics()
    active = UMI_ACTIVE_DIMS
    if np.any(extrema_min[active] < statistics["actions_min"][active] - 1e-6) or np.any(
        extrema_max[active] > statistics["actions_max"][active] + 1e-6
    ):
        raise ValueError("sampled q0 chunks exceed fixed normalization bounds")
    print(json.dumps({
        "episodes": len(files), "splits": split_counts, "frames": total_frames,
        "action_dim": SHARED_ACTION_DIM, "active_action_dims": len(active),
        "status": "ready_for_training",
    }, indent=2))


if __name__ == "__main__":
    main()

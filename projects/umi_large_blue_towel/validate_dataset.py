#!/usr/bin/env python3
"""Fail-fast validation for the converted UMI Cosmos dataset."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import cv2
import h5py
import numpy as np

TASK = "fold the large blue towel twice"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()
    files = sorted((args.data_dir / "train").glob("episode_*.hdf5"))
    if len(files) != 48:
        raise ValueError(f"expected 48 HDF5 episodes, found {len(files)}")
    total_frames = 0
    max_step = 0.0
    for path in files:
        with h5py.File(path, "r") as root:
            action = root["action"][:]
            proprio = root["observations/qpos"][:]
            if action.ndim != 2 or action.shape[1] != 29 or proprio.shape != action.shape:
                raise ValueError(f"bad tensors in {path}: {action.shape}/{proprio.shape}")
            if not np.isfinite(action).all() or not np.isfinite(proprio).all():
                raise ValueError(f"non-finite tensor in {path}")
            if np.count_nonzero(action[:, :17]) or np.count_nonzero(proprio[:, :17]):
                raise ValueError(f"unavailable channels are not zero in {path}")
            if root.attrs["task_description"] != TASK:
                raise ValueError(f"bad task text in {path}")
            total_frames += len(action)
            max_step = max(max_step, float(np.linalg.norm(np.diff(action[:, 17:], axis=0), axis=1).max()))
            for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                name = root["observations/video_paths"][camera][()].decode()
                video = os.path.join(path.parent, name)
                cap = cv2.VideoCapture(video)
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if count != len(action) or abs(fps - 25.0) > 0.01:
                    raise ValueError(f"video mismatch {video}: frames={count}, fps={fps}, actions={len(action)}")
    stats = json.loads((args.data_dir / "dataset_statistics.json").read_text())
    for key in ("actions", "proprio"):
        minimum = np.asarray(stats[f"{key}_min"])
        maximum = np.asarray(stats[f"{key}_max"])
        if minimum.shape != (29,) or maximum.shape != (29,):
            raise ValueError(f"bad {key} stats shape")
        if np.any(maximum <= minimum) or not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
            raise ValueError(f"invalid {key} normalization range")
    with (args.data_dir / "t5_embeddings.pkl").open("rb") as stream:
        embeddings = pickle.load(stream)
    if TASK not in embeddings or tuple(embeddings[TASK].shape) != (1, 512, 1024):
        raise ValueError("missing or malformed task embedding")
    manifest = json.loads((args.data_dir / "conversion_manifest.json").read_text())
    if len(manifest["episodes"]) != 48 or manifest["total_frames"] != total_frames:
        raise ValueError("conversion manifest mismatch")
    print(f"VALID: episodes=48 frames={total_frames} hours={total_frames / 25 / 3600:.3f} max_ee_step={max_step:.4f}")


if __name__ == "__main__":
    main()

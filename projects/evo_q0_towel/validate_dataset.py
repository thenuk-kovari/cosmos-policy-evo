#!/usr/bin/env python3
"""Fail-fast validation of converted EVO q0 episodes and action statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from cosmos_policy.datasets.evo_q0_actions import (
    ACTION_DIM,
    CHUNK_SIZE,
    CONTRACT_NAME,
    build_q0_anchored_chunk,
    load_or_compute_q0_action_statistics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()
    files = sorted((args.data_dir / "train").glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"no HDF5 episodes in {args.data_dir / 'train'}")

    trajectories = []
    total_frames = 0
    for path in files:
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("action_contract") != CONTRACT_NAME:
                raise ValueError(f"{path}: wrong or missing action contract")
            state = handle["observations/qpos"][:]
            source = handle["action"][:]
            if state.shape != source.shape or state.ndim != 2 or state.shape[1] != ACTION_DIM:
                raise ValueError(f"{path}: expected identical [T,17] state/action source")
            if not np.array_equal(state, source):
                raise ValueError(f"{path}: action contains commands or deltas instead of measured state")
            if len(state) < CHUNK_SIZE + 1 or not np.isfinite(state).all():
                raise ValueError(f"{path}: invalid trajectory length or non-finite state")
            for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                value = handle[f"observations/video_paths/{camera}"][()]
                filename = value.decode() if isinstance(value, bytes) else str(value)
                if not (path.parent / filename).is_file():
                    raise FileNotFoundError(path.parent / filename)
            trajectories.append(state)
            total_frames += len(state)

    stats = load_or_compute_q0_action_statistics(args.data_dir, trajectories)
    probes = [build_q0_anchored_chunk(x, min(len(x) // 2, len(x) - 1)) for x in trajectories]
    report = {
        "contract": CONTRACT_NAME,
        "episodes": len(files),
        "frames": total_frames,
        "chunk_shape": list(probes[0].shape),
        "action_min": stats["actions_min"].tolist(),
        "action_max": stats["actions_max"].tolist(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

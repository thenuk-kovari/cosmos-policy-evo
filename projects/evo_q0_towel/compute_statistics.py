#!/usr/bin/env python3
"""Compute one normalization contract across one or more EVO datasets."""

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
    calculate_q0_action_statistics,
)


def vector_statistics(values: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_min": values.min(axis=0).astype(np.float32),
        f"{prefix}_max": values.max(axis=0).astype(np.float32),
        f"{prefix}_mean": values.mean(axis=0).astype(np.float32),
        f"{prefix}_std": values.std(axis=0).astype(np.float32),
        f"{prefix}_median": np.median(values, axis=0).astype(np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    trajectories = []
    files = []
    for data_dir in args.data_dir:
        for path in sorted((data_dir / "train").glob("*.hdf5")):
            with h5py.File(path, "r") as handle:
                if handle.attrs.get("action_contract") != CONTRACT_NAME:
                    raise ValueError(f"{path}: wrong or missing action_contract")
                state = handle["observations/qpos"][:].astype(np.float32)
                source = handle["action"][:].astype(np.float32)
                if state.shape != source.shape or state.ndim != 2 or state.shape[1] != ACTION_DIM:
                    raise ValueError(f"{path}: expected identical [T,17] state/action source")
                if not np.array_equal(state, source):
                    raise ValueError(f"{path}: action source is not measured state")
                trajectories.append(state)
                files.append(str(path))
    if not trajectories:
        raise FileNotFoundError("no valid HDF5 episodes found")

    action_stats = calculate_q0_action_statistics(trajectories, chunk_size=CHUNK_SIZE)
    proprio_stats = vector_statistics(np.concatenate(trajectories, axis=0), "proprio")
    statistics = {**action_stats, **proprio_stats}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "dataset_statistics.json").write_text(
        json.dumps({key: value.tolist() for key, value in statistics.items()}, indent=2) + "\n"
    )
    (args.out / "q0_action_statistics.json").write_text(
        json.dumps(
            {
                "contract": CONTRACT_NAME,
                "chunk_size": CHUNK_SIZE,
                "first_future_offset": 1,
                "gripper_mode": "absolute_measured_joint_position_rad",
                "statistics": {key: value.tolist() for key, value in action_stats.items()},
                "source_files": files,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"episodes": len(files), "frames": sum(map(len, trajectories)), "out": str(args.out)}))


if __name__ == "__main__":
    main()

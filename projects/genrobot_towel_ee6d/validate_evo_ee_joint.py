#!/usr/bin/env python3
"""Fail-fast validation for the Evo shared EE + joint q0 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

from cosmos_policy.datasets.ee_q0_actions import (
    CONTRACT_NAME,
    EVO_STORAGE_CONTRACT,
    NORMALIZATION_CONTRACT,
    SHARED_ACTION_DIM,
    build_evo_shared_action_chunk_from_storage,
    canonical_shared_statistics,
    normalize_shared_action_chunk,
    rotation_6d_to_matrix,
    LEFT_EE_ROTATION_6D,
    RIGHT_EE_ROTATION_6D,
)
from projects.genrobot_towel_ee6d.convert_evo_fk import FK, OPEN_RAD, convert_states


def attr(handle: h5py.File, name: str) -> str:
    value = handle.attrs.get(name, "")
    return value.decode() if isinstance(value, bytes) else str(value)


def video_frames(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open video {path}")
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=73)
    args = parser.parse_args()

    manifest = json.loads((args.dataset / "conversion_manifest.json").read_text())
    expected_contract = {
        "action_contract": CONTRACT_NAME,
        "storage_contract": EVO_STORAGE_CONTRACT,
        "normalization_contract": NORMALIZATION_CONTRACT,
        "action_dim": SHARED_ACTION_DIM,
        "proprio_dim": 17,
        "chunk_size": 50,
        "recorded_action_column_used": False,
    }
    for key, expected in expected_contract.items():
        if manifest.get(key) != expected:
            raise ValueError(f"manifest {key}={manifest.get(key)!r}, expected {expected!r}")
    if manifest["counts"]["total"] != args.expected_episodes:
        raise ValueError(f"manifest has {manifest['counts']['total']} episodes")

    statistics = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in json.loads((args.dataset / "dataset_statistics.json").read_text()).items()
    }
    for key, expected in canonical_shared_statistics().items():
        if key not in statistics or not np.allclose(statistics[key], expected, atol=1e-7):
            raise ValueError(f"statistics mismatch for {key}")

    files = sorted(args.dataset.glob("train/*.hdf5")) + sorted(args.dataset.glob("val/*.hdf5"))
    if len(files) != args.expected_episodes:
        raise ValueError(f"found {len(files)} HDF5 files, expected {args.expected_episodes}")
    fk = FK(args.urdf)
    total_frames = 0
    normalized_min = np.full(SHARED_ACTION_DIM, np.inf)
    normalized_max = np.full(SHARED_ACTION_DIM, -np.inf)
    for index, path in enumerate(files, 1):
        with h5py.File(path, "r") as handle:
            expected_attrs = {
                "action_contract": CONTRACT_NAME,
                "action_storage_contract": EVO_STORAGE_CONTRACT,
                "normalization_contract": NORMALIZATION_CONTRACT,
                "source_pose_link": "base_link",
                "fk_root": "base_link",
                "fk_tips": "left_palm,right_palm",
                "position_post_transform": "none",
                "orientation_post_transform": "right_multiply_q_offset_wxyz",
                "delta_reference_frame": "query_body",
                "recorded_action_column_used": "False",
                "task_description": "fold the blue towel twice",
            }
            for name, expected in expected_attrs.items():
                if attr(handle, name) != expected:
                    raise ValueError(f"{path}: {name}={attr(handle, name)!r}, expected {expected!r}")
            source = handle["action"][:]
            proprio = handle["observations/qpos"][:]
            if source.shape != (len(proprio), SHARED_ACTION_DIM) or proprio.shape[1] != 17:
                raise ValueError(f"{path}: wrong action/proprio shapes {source.shape}, {proprio.shape}")
            if len(source) < 51 or not np.isfinite(source).all() or not np.isfinite(proprio).all():
                raise ValueError(f"{path}: short or non-finite trajectory")
            if not np.array_equal(handle["action_dim_mask"][:], np.ones(35, np.float32)):
                raise ValueError(f"{path}: wrong action mask")
            if not np.array_equal(handle["proprio_dim_mask"][:], np.ones(17, np.float32)):
                raise ValueError(f"{path}: wrong proprio mask")
            if np.any((proprio[:, [7, 15]] < 0) | (proprio[:, [7, 15]] > 1)):
                raise ValueError(f"{path}: canonical gripper outside [0,1]")
            rotation_6d_to_matrix(source[:, LEFT_EE_ROTATION_6D])
            rotation_6d_to_matrix(source[:, RIGHT_EE_ROTATION_6D])

            # Independently recompute FK on beginning/middle/end frames from
            # stored proprio and require exact agreement with stored action.
            selected = np.unique([0, len(proprio) // 2, len(proprio) - 1])
            raw = proprio[selected].copy()
            raw[:, 7] *= OPEN_RAD
            raw[:, 15] *= OPEN_RAD
            expected_source, expected_proprio = convert_states(raw, fk)
            if not np.allclose(source[selected], expected_source, atol=2e-5):
                raise ValueError(f"{path}: base_link/palm FK source mismatch")
            if not np.allclose(proprio[selected], expected_proprio, atol=2e-6):
                raise ValueError(f"{path}: proprio canonicalization mismatch")

            # Check representative q0 anchors, terminal padding, and fixed
            # physical normalization bounds.
            anchors = np.unique([0, len(source) // 3, 2 * len(source) // 3, len(source) - 1])
            for anchor in anchors:
                chunk, mask = build_evo_shared_action_chunk_from_storage(source, int(anchor), 50)
                if not np.array_equal(mask, np.ones(35, np.float32)):
                    raise ValueError(f"{path}: constructed action mask mismatch")
                normalized = normalize_shared_action_chunk(chunk, statistics)
                if np.any(normalized < -1.0001) or np.any(normalized > 1.0001):
                    raise ValueError(f"{path}: action exceeds fixed normalization bounds at anchor {anchor}")
                normalized_min = np.minimum(normalized_min, normalized.min(axis=0))
                normalized_max = np.maximum(normalized_max, normalized.max(axis=0))
            if not np.allclose(build_evo_shared_action_chunk_from_storage(source, len(source) - 1)[0][:, :3], 0):
                raise ValueError(f"{path}: terminal q0 translation padding is not zero")

            for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                value = handle["observations/video_paths"][camera][()]
                filename = value.decode() if isinstance(value, bytes) else str(value)
                if video_frames(path.parent / filename) != len(source):
                    raise ValueError(f"{path}: {camera} frame count mismatch")
            total_frames += len(source)
        print(f"[{index}/{len(files)}] valid {path}", flush=True)

    if total_frames != manifest["total_frames"]:
        raise ValueError(f"frame count {total_frames} != manifest {manifest['total_frames']}")
    result = {
        "episodes": len(files),
        "frames": total_frames,
        "action_dim": 35,
        "proprio_dim": 17,
        "contract": CONTRACT_NAME,
        "storage_contract": EVO_STORAGE_CONTRACT,
        "normalized_min": normalized_min.tolist(),
        "normalized_max": normalized_max.tolist(),
    }
    (args.dataset / "validation_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert GenRobot fold-towel MCAPs into Cosmos q0-local EE-6D training data."""

from __future__ import annotations

import argparse
import fcntl
import os
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import h5py
import imageio_ffmpeg
import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory
from scipy.spatial.transform import Rotation, Slerp

from cosmos_policy.datasets.ee_q0_actions import (
    CONTRACT_NAME,
    STORAGE_CONTRACT,
    NORMALIZATION_CONTRACT,
    LEFT_GRIPPER,
    RIGHT_GRIPPER,
    SHARED_ACTION_DIM,
    UMI_ACTIVE_DIMS,
    build_shared_pose_source,
    canonical_shared_statistics,
)
from projects.genrobot_towel_ee6d.audit_mcap import inspect


FPS = 30.0
PROPRIO_DIM = 17
DEFAULT_TASK = "fold the towel"
DAS_MAX_GRIPPER_WIDTH_M = 0.103


def timestamp_ns(decoded, fallback: int) -> int:
    header = getattr(decoded, "header", None)
    value = int(getattr(header, "timestamp", 0)) if header is not None else 0
    return value or int(fallback)


def read_streams(path: Path) -> dict:
    values = {
        robot: defaultdict(list)
        for robot in (0, 1)
    }
    with path.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _, channel, message, decoded in reader.iter_decoded_messages():
            topic = channel.topic
            for robot in (0, 1):
                prefix = f"/robot{robot}/"
                if not topic.startswith(prefix):
                    continue
                ts = timestamp_ns(decoded, message.log_time)
                if topic == prefix + "vio/eef_pose":
                    p, q = decoded.pose.position, decoded.pose.orientation
                    values[robot]["pose_ts"].append(ts)
                    values[robot]["position"].append((p.x, p.y, p.z))
                    values[robot]["quaternion"].append((q.x, q.y, q.z, q.w))
                    values[robot]["pose_frame"].append(str(decoded.frame_id))
                elif topic == prefix + "sensor/magnetic_encoder":
                    values[robot]["gripper_ts"].append(ts)
                    values[robot]["gripper"].append(float(decoded.value))
                elif topic == prefix + "sensor/camera0/compressed":
                    if decoded.format != "h264":
                        raise ValueError(f"{path}: expected h264, got {decoded.format!r}")
                    values[robot]["camera_ts"].append(ts)
                    values[robot]["camera_packet"].append(bytes(decoded.data))
                    values[robot]["camera_frame"].append(str(decoded.frame_id))
    return values


def unique_sorted(timestamps: list[int], *arrays: list) -> tuple[np.ndarray, ...]:
    ts = np.asarray(timestamps, dtype=np.int64)
    order = np.argsort(ts, kind="stable")
    ts = ts[order]
    keep = np.r_[True, np.diff(ts) > 0]
    result = [ts[keep]]
    for value in arrays:
        result.append(np.asarray(value)[order][keep])
    return tuple(result)


def interpolate_pose(values: dict, target_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ts, position, quaternion = unique_sorted(
        values["pose_ts"], values["position"], values["quaternion"]
    )
    relative_s = (ts - ts[0]).astype(np.float64) / 1e9
    target_s = (target_ns - ts[0]).astype(np.float64) / 1e9
    out_position = np.stack(
        [np.interp(target_s, relative_s, position[:, axis]) for axis in range(3)], axis=1
    )
    out_quaternion = Slerp(relative_s, Rotation.from_quat(quaternion))(target_s).as_quat()
    return out_position, out_quaternion


def interpolate_scalar(timestamps: list[int], values: list[float], target_ns: np.ndarray) -> np.ndarray:
    ts, scalar = unique_sorted(timestamps, values)
    relative_s = (ts - ts[0]).astype(np.float64) / 1e9
    target_s = (target_ns - ts[0]).astype(np.float64) / 1e9
    return np.interp(target_s, relative_s, scalar)


def nearest_indices(timestamps: list[int], target_ns: np.ndarray) -> tuple[np.ndarray, float]:
    ts = np.asarray(timestamps, dtype=np.int64)
    if np.any(np.diff(ts) <= 0):
        raise ValueError("camera timestamps must be strictly increasing")
    upper = np.searchsorted(ts, target_ns, side="left")
    upper = np.clip(upper, 0, len(ts) - 1)
    lower = np.maximum(upper - 1, 0)
    choose_upper = np.abs(ts[upper] - target_ns) < np.abs(ts[lower] - target_ns)
    indices = np.where(choose_upper, upper, lower)
    max_error_ms = float(np.abs(ts[indices] - target_ns).max(initial=0) / 1e6)
    return indices, max_error_ms


def apply_base_to_action_frame(
    position: np.ndarray, quaternion_xyzw: np.ndarray, transform: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compose world_T_base with the configured constant base_T_action."""
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (7,):
        raise ValueError("base-to-action transform must be tx ty tz qx qy qz qw")
    base_rotation = Rotation.from_quat(quaternion_xyzw).as_matrix()
    fixed_rotation = Rotation.from_quat(transform[3:]).as_matrix()
    action_position = position + np.einsum("tij,j->ti", base_rotation, transform[:3])
    action_rotation = np.einsum("tij,jk->tik", base_rotation, fixed_rotation)
    return action_position, Rotation.from_matrix(action_rotation).as_quat()


def write_selected_video(
    packets: list[bytes], selected_indices: np.ndarray, output: Path, fps: float
) -> tuple[int, tuple[int, int]]:
    if len(selected_indices) == 0 or np.any(np.diff(selected_indices) < 0):
        raise ValueError("selected camera indices must be non-empty and monotonic")
    with tempfile.TemporaryDirectory(prefix="genrobot_h264_") as directory:
        elementary = Path(directory) / "input.h264"
        with elementary.open("wb") as stream:
            for packet in packets:
                stream.write(packet)
        reader = imageio_ffmpeg.read_frames(str(elementary), pix_fmt="rgb24", output_params=["-vsync", "0"])
        metadata = next(reader)
        size = tuple(metadata["size"])
        writer = imageio_ffmpeg.write_frames(
            str(output),
            size,
            fps=fps,
            codec="libx264",
            quality=7,
            macro_block_size=2,
            output_params=["-preset", "fast", "-g", str(round(fps))],
        )
        writer.send(None)
        wanted = iter(enumerate(selected_indices.tolist()))
        next_wanted = next(wanted, None)
        written = 0
        try:
            for source_index, frame in enumerate(reader):
                while next_wanted is not None and next_wanted[1] == source_index:
                    writer.send(frame)
                    written += 1
                    next_wanted = next(wanted, None)
                if next_wanted is None:
                    break
        finally:
            reader.close()
            writer.close()
        if next_wanted is not None or written != len(selected_indices):
            raise ValueError(f"decoded {written}/{len(selected_indices)} selected frames")
        return written, size


def write_black_video(output: Path, frames: int, fps: float, size: tuple[int, int] = (224, 224)) -> None:
    writer = imageio_ffmpeg.write_frames(
        str(output), size, fps=fps, codec="libx264", quality=7,
        output_params=["-preset", "veryfast", "-g", str(round(fps))],
    )
    writer.send(None)
    black = bytes(size[0] * size[1] * 3)
    try:
        for _ in range(frames):
            writer.send(black)
    finally:
        writer.close()


def write_statistics(root: Path) -> None:
    statistics = canonical_shared_statistics()
    payload = {key: value.tolist() for key, value in statistics.items()}
    path = root / "dataset_statistics.json"
    contract = {
        "name": NORMALIZATION_CONTRACT,
        "action_dim": SHARED_ACTION_DIM,
        "proprio_dim": PROPRIO_DIM,
        "method": "fixed physical min-max to [-1,1]",
        "statistics": payload,
    }
    with (root / ".metadata.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists() and json.loads(path.read_text()) != payload:
            raise ValueError(f"refusing to replace incompatible statistics: {path}")
        path.write_text(json.dumps(payload, indent=2) + "\n")
        (root / "normalization_contract.json").write_text(json.dumps(contract, indent=2) + "\n")


def convert_one(
    source_path: Path,
    output_root: Path,
    task: str,
    val_every: int,
    left_base_to_action: np.ndarray,
    right_base_to_action: np.ndarray,
) -> dict:
    audit = inspect(source_path)
    if not audit["accepted"]:
        return {"source": str(source_path), "accepted": False, "checks": audit["checks"]}
    streams = read_streams(source_path)
    dynamic_keys = ("pose_ts", "gripper_ts", "camera_ts")
    overlap_start = max(min(streams[robot][key]) for robot in (0, 1) for key in dynamic_keys)
    overlap_end = min(max(streams[robot][key]) for robot in (0, 1) for key in dynamic_keys)
    left_camera_ts = np.asarray(streams[0]["camera_ts"], dtype=np.int64)
    left_indices = np.flatnonzero((left_camera_ts >= overlap_start) & (left_camera_ts <= overlap_end))
    target_ns = left_camera_ts[left_indices]
    right_indices, right_error_ms = nearest_indices(streams[1]["camera_ts"], target_ns)
    if right_error_ms > 25.0:
        return {"source": str(source_path), "accepted": False, "checks": [f"camera_sync_{right_error_ms:.2f}ms"]}
    if len(target_ns) < 2 * 50:
        return {"source": str(source_path), "accepted": False, "checks": ["fewer_than_100_frames"]}

    left_position, left_quaternion = interpolate_pose(streams[0], target_ns)
    right_position, right_quaternion = interpolate_pose(streams[1], target_ns)
    left_position, left_quaternion = apply_base_to_action_frame(
        left_position, left_quaternion, left_base_to_action
    )
    right_position, right_quaternion = apply_base_to_action_frame(
        right_position, right_quaternion, right_base_to_action
    )
    left_gripper_m = interpolate_scalar(streams[0]["gripper_ts"], streams[0]["gripper"], target_ns)
    right_gripper_m = interpolate_scalar(streams[1]["gripper_ts"], streams[1]["gripper"], target_ns)
    action_source = build_shared_pose_source(
        left_position, left_quaternion, left_gripper_m,
        right_position, right_quaternion, right_gripper_m,
    )
    proprio = np.zeros((len(target_ns), PROPRIO_DIM), dtype=np.float32)
    proprio[:, 7] = action_source[:, LEFT_GRIPPER]
    proprio[:, 15] = action_source[:, RIGHT_GRIPPER]
    mask = np.zeros(SHARED_ACTION_DIM, dtype=np.float32)
    mask[UMI_ACTIVE_DIMS] = 1.0
    proprio_mask = np.zeros(PROPRIO_DIM, dtype=np.float32)
    proprio_mask[[7, 15]] = 1.0

    episode_number = int(source_path.stem)
    split = "val" if val_every > 0 and episode_number % val_every == 0 else "train"
    output_dir = output_root / split
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"episode_{source_path.stem}"
    left_video = output_dir / f"{stem}_left.mp4"
    right_video = output_dir / f"{stem}_right.mp4"
    high_video = output_dir / f"{stem}_high_black.mp4"
    hdf5_path = output_dir / f"{stem}.hdf5"
    for path in (left_video, right_video, high_video, hdf5_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    left_written, left_size = write_selected_video(
        streams[0]["camera_packet"], left_indices, left_video, FPS
    )
    right_written, right_size = write_selected_video(
        streams[1]["camera_packet"], right_indices, right_video, FPS
    )
    write_black_video(high_video, len(target_ns), FPS)
    if left_written != len(target_ns) or right_written != len(target_ns):
        raise RuntimeError("video and state lengths differ")

    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(hdf5_path, "w") as handle:
        handle.attrs["task_description"] = task
        handle.attrs["action_contract"] = CONTRACT_NAME
        handle.attrs["action_storage_contract"] = STORAGE_CONTRACT
        handle.attrs["normalization_contract"] = NORMALIZATION_CONTRACT
        handle.attrs["source_pose_link"] = "base_link"
        handle.attrs["source_world_frame"] = "world"
        handle.attrs["delta_reference_frame"] = "query_body"
        handle.attrs["rotation_6d_layout"] = "first_two_columns_c0_then_c1"
        handle.attrs["left_source_robot"] = 0
        handle.attrs["right_source_robot"] = 1
        handle.attrs["left_base_to_action_xyzw"] = left_base_to_action
        handle.attrs["right_base_to_action_xyzw"] = right_base_to_action
        handle.attrs["fps"] = FPS
        handle.create_dataset("action", data=action_source, compression="gzip")
        handle.create_dataset("relative_action", data=action_source, compression="gzip")
        handle.create_dataset("action_dim_mask", data=mask)
        handle.create_dataset("proprio_dim_mask", data=proprio_mask)
        observations = handle.create_group("observations")
        observations.create_dataset("qpos", data=proprio, compression="gzip")
        video_paths = observations.create_group("video_paths")
        video_paths.create_dataset("cam_high", data=high_video.name, dtype=string_dtype)
        video_paths.create_dataset("cam_left_wrist", data=left_video.name, dtype=string_dtype)
        video_paths.create_dataset("cam_right_wrist", data=right_video.name, dtype=string_dtype)

    return {
        "source": str(source_path), "accepted": True, "split": split,
        "frames": len(target_ns), "duration_s": len(target_ns) / FPS,
        "max_right_camera_sync_error_ms": right_error_ms,
        "left_source_size": left_size, "right_source_size": right_size,
        "hdf5": str(hdf5_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mcaps", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--val-every", type=int, default=10, help="every Nth numeric episode is validation")
    identity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    parser.add_argument("--left-base-to-action", type=float, nargs=7, default=identity, metavar=("TX","TY","TZ","QX","QY","QZ","QW"))
    parser.add_argument("--right-base-to-action", type=float, nargs=7, default=identity, metavar=("TX","TY","TZ","QX","QY","QZ","QW"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    write_statistics(args.output)
    reports = [
        convert_one(
            path, args.output, args.task, args.val_every,
            np.asarray(args.left_base_to_action), np.asarray(args.right_base_to_action),
        )
        for path in args.mcaps
    ]
    manifest_path = args.output / "conversion_manifest.json"
    with (args.output / ".manifest.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
        temporary = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(previous + reports, indent=2) + "\n")
        temporary.replace(manifest_path)
    print(json.dumps({
        "accepted": sum(report["accepted"] for report in reports),
        "rejected": sum(not report["accepted"] for report in reports),
        "reports": reports,
    }, indent=2))
    raise SystemExit(0 if all(report["accepted"] for report in reports) else 2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert reviewed YUBI UMI LeRobot Parquet episodes to Cosmos ALOHA format."""

from __future__ import annotations

import argparse
import io
import json
import shutil
from pathlib import Path

import h5py
import imageio_ffmpeg
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from scipy.spatial.transform import Rotation

TASK = "fold the large blue towel twice"
SOURCE_FPS = 30.0
TARGET_FPS = 25
ACTION_DIM = 29
EE_START = 17
CAMERAS = {
    "cam_high": "observation.images.base",
    "cam_left_wrist": "observation.images.left_gripper",
    "cam_right_wrist": "observation.images.right_gripper",
}


def continuous_quaternions_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("zero-norm quaternion")
    quat /= norms
    for i in range(1, len(quat)):
        if np.dot(quat[i - 1], quat[i]) < 0:
            quat[i] *= -1
    return quat


def continuous_rotvec(quat: np.ndarray) -> np.ndarray:
    raw = Rotation.from_quat(continuous_quaternions_xyzw(quat)).as_rotvec()
    result = raw.copy()
    for i in range(1, len(raw)):
        norm = np.linalg.norm(raw[i])
        if norm < 1e-8:
            continue
        axis = raw[i] / norm
        candidates = np.stack([raw[i] + 2 * np.pi * k * axis for k in range(-2, 3)])
        result[i] = candidates[np.argmin(np.linalg.norm(candidates - result[i - 1], axis=1))]
    return result.astype(np.float32)


def pose16_to_ee12(values: np.ndarray) -> np.ndarray:
    """[left xyz,xyzw,grip,right xyz,xyzw,grip] -> 2x[xyz,rotvec]."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 16:
        raise ValueError(f"expected (T,16), got {values.shape}")
    return np.concatenate(
        [
            values[:, 0:3],
            continuous_rotvec(values[:, 3:7]),
            values[:, 8:11],
            continuous_rotvec(values[:, 11:15]),
        ],
        axis=1,
    ).astype(np.float32)


def full29(ee12: np.ndarray) -> np.ndarray:
    result = np.zeros((len(ee12), ACTION_DIM), dtype=np.float32)
    result[:, EE_START:] = ee12
    return result


def nearest_indices(timestamps: np.ndarray, start: float, end: float) -> np.ndarray:
    if end <= start:
        raise ValueError(f"invalid trim [{start}, {end}]")
    target = start + np.arange(int(np.floor((end - start) * TARGET_FPS)), dtype=np.float64) / TARGET_FPS
    pos = np.searchsorted(timestamps, target, side="left")
    pos = np.clip(pos, 0, len(timestamps) - 1)
    prev = np.maximum(pos - 1, 0)
    use_prev = np.abs(timestamps[prev] - target) <= np.abs(timestamps[pos] - target)
    return np.where(use_prev, prev, pos).astype(np.int64)


def decode_resize(item: dict, size: int) -> np.ndarray:
    with Image.open(io.BytesIO(item["bytes"])) as image:
        image = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        return np.asarray(image, dtype=np.uint8)


def write_video(path: Path, items: list[dict], indices: np.ndarray, size: int) -> None:
    writer = imageio_ffmpeg.write_frames(
        str(path),
        (size, size),
        fps=TARGET_FPS,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-crf", "23", "-movflags", "+faststart"],
        macro_block_size=1,
    )
    writer.send(None)
    try:
        for idx in indices:
            writer.send(np.ascontiguousarray(decode_resize(items[int(idx)], size)))
    finally:
        writer.close()


def write_hdf5(path: Path, action: np.ndarray, proprio: np.ndarray, videos: dict[str, str], source: str) -> None:
    qvel = np.gradient(proprio, 1.0 / TARGET_FPS, axis=0).astype(np.float32)
    relative = np.zeros_like(action)
    relative[:-1] = action[1:] - action[:-1]
    relative[-1] = relative[-2]
    with h5py.File(path, "w") as root:
        root.attrs["sim"] = False
        root.attrs["success"] = True
        root.attrs["task_description"] = TASK
        root.attrs["source_episode"] = source
        root.attrs["fps"] = TARGET_FPS
        root.attrs["action_layout"] = "joint_gripper16,elevator1,left_ee_xyz_rotvec6,right_ee_xyz_rotvec6"
        obs = root.create_group("observations")
        obs.create_dataset("qpos", data=proprio)
        obs.create_dataset("qvel", data=qvel)
        obs.create_dataset("effort", data=np.zeros_like(proprio))
        video_paths = obs.create_group("video_paths")
        for camera, name in videos.items():
            video_paths.create_dataset(camera, data=name.encode("utf-8"))
        root.create_dataset("action", data=action)
        root.create_dataset("relative_action", data=relative)


def stats(actions: list[np.ndarray], proprios: list[np.ndarray], fill_missing_ranges: bool = True) -> dict[str, list[float]]:
    result = {}
    for key, arrays in (("actions", actions), ("proprio", proprios)):
        values = np.concatenate(arrays, axis=0)
        minimum = values.min(axis=0)
        maximum = values.max(axis=0)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        median = np.median(values, axis=0)
        if fill_missing_ranges:
            # Missing UMI embodiment channels normalize to exactly zero.
            minimum[:EE_START] = -1.0
            maximum[:EE_START] = 1.0
            mean[:EE_START] = 0.0
            std[:EE_START] = 1.0
            median[:EE_START] = 0.0
        result.update(
            {
                f"{key}_min": minimum.tolist(),
                f"{key}_max": maximum.tolist(),
                f"{key}_mean": mean.tolist(),
                f"{key}_std": std.tolist(),
                f"{key}_median": median.tolist(),
            }
        )
    return result


def convert(args: argparse.Namespace) -> None:
    reviews = json.loads(args.reviews.read_text())
    kept = [i for i in sorted(map(int, reviews)) if reviews[str(i)]["verdict"] == "keep"]
    if len(reviews) != 50 or len(kept) != 48:
        raise ValueError(f"expected 50 reviewed/48 kept, got {len(reviews)}/{len(kept)}")
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(args.output)
        shutil.rmtree(args.output)
    train = args.output / "train"
    train.mkdir(parents=True)
    all_actions, all_proprios, manifest = [], [], []
    columns = ["observation.state", "action", "timestamp", *CAMERAS.values()]
    for output_idx, episode in enumerate(kept):
        source = args.source / "data" / "chunk-000" / f"file-{episode + 1:03d}.parquet"
        if not source.exists():
            raise FileNotFoundError(source)
        print(f"[{output_idx + 1}/{len(kept)}] episode {episode}: {source.name}", flush=True)
        table = pq.read_table(source, columns=columns)
        timestamps = np.asarray(table["timestamp"], dtype=np.float64)
        review = reviews[str(episode)]
        indices = nearest_indices(timestamps, float(review["trim_start"]), float(review["trim_end"]))
        if len(indices) < 51:
            raise ValueError(f"episode {episode} too short after trim: {len(indices)}")
        source_action = np.asarray(table["action"].to_pylist(), dtype=np.float32)[indices]
        source_state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)[indices]
        action = full29(pose16_to_ee12(source_action))
        proprio = full29(pose16_to_ee12(source_state))
        if not np.isfinite(action).all() or not np.isfinite(proprio).all():
            raise ValueError(f"non-finite values in episode {episode}")
        stem = f"episode_{output_idx}"
        videos = {}
        for camera, column in CAMERAS.items():
            name = f"{stem}_{camera}.mp4"
            videos[camera] = name
            write_video(train / name, table[column].to_pylist(), indices, args.image_size)
        write_hdf5(train / f"{stem}.hdf5", action, proprio, videos, source.name)
        all_actions.append(action)
        all_proprios.append(proprio)
        manifest.append(
            {
                "source_episode": episode,
                "source_file": source.name,
                "output_stem": stem,
                "trim_start": review["trim_start"],
                "trim_end": review["trim_end"],
                "frames": len(indices),
            }
        )
    pre_stats = stats(all_actions, all_proprios)
    (args.output / "dataset_statistics.json").write_text(json.dumps(pre_stats, indent=2))
    normalized = []
    for key, arrays in (("actions", all_actions), ("proprio", all_proprios)):
        minimum = np.asarray(pre_stats[f"{key}_min"], dtype=np.float32)
        maximum = np.asarray(pre_stats[f"{key}_max"], dtype=np.float32)
        normalized.append([2 * ((array - minimum) / (maximum - minimum)) - 1 for array in arrays])
    post_stats = stats(normalized[0], normalized[1], fill_missing_ranges=False)
    (args.output / "dataset_statistics_post_norm.json").write_text(json.dumps(post_stats, indent=2))
    summary = {
        "task": TASK,
        "source_fps": SOURCE_FPS,
        "target_fps": TARGET_FPS,
        "action_dim": ACTION_DIM,
        "action_mask": [0] * EE_START + [1] * (ACTION_DIM - EE_START),
        "action_semantics": "absolute xyz+rotation-vector per arm in stored source frame",
        "episodes": manifest,
        "total_frames": sum(item["frames"] for item in manifest),
    }
    (args.output / "conversion_manifest.json").write_text(json.dumps(summary, indent=2))
    print(f"ready: {args.output} ({summary['total_frames']} frames)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    convert(parser.parse_args())


if __name__ == "__main__":
    main()

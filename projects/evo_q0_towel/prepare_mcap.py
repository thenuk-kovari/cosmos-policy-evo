#!/usr/bin/env python3
"""Convert EVO MCAP demonstrations to the q0-observed-state Cosmos format."""

from __future__ import annotations

import argparse
import io
import json
import shutil
from pathlib import Path

import h5py
import imageio_ffmpeg
import numpy as np
from PIL import Image
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

from cosmos_policy.datasets.evo_q0_actions import ACTION_DIM, ACTION_LAYOUT, CONTRACT_NAME


DEFAULT_FPS = 30
DEFAULT_TASK = "fold the blue towel twice"
JOINT_NAMES = (
    *(f"left_joint{i}" for i in range(1, 8)),
    "left_finger_joint1",
    *(f"right_joint{i}" for i in range(1, 8)),
    "right_finger_joint1",
    "carriage_joint",
)
CAMERA_TOPICS = {
    "cam_high": ("/base/image_raw", "/base/image_raw/compressed"),
    "cam_left_wrist": ("/left_arm/image_raw", "/left_arm/image_raw/compressed"),
    "cam_right_wrist": ("/right_arm/image_raw", "/right_arm/image_raw/compressed"),
}
ALL_TOPICS = {"/joint_states", *(topic for choices in CAMERA_TOPICS.values() for topic in choices)}


def nearest_indices(timestamps: np.ndarray, query: np.ndarray) -> np.ndarray:
    right = np.searchsorted(timestamps, query).clip(0, len(timestamps) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(np.abs(timestamps[left] - query) <= np.abs(timestamps[right] - query), left, right)


def read_mcap(path: Path) -> dict[str, list[tuple[float, object]]]:
    streams = {topic: [] for topic in ALL_TOPICS}
    with path.open("rb") as handle:
        reader = make_reader(handle, decoder_factories=[DecoderFactory()])
        for _, channel, message, decoded in reader.iter_decoded_messages():
            if channel.topic in streams:
                streams[channel.topic].append((message.log_time * 1e-9, decoded))
    for rows in streams.values():
        rows.sort(key=lambda row: row[0])
    return streams


def select_camera_stream(
    streams: dict[str, list[tuple[float, object]]], camera_name: str
) -> tuple[str, list[tuple[float, object]]]:
    present = [(topic, streams[topic]) for topic in CAMERA_TOPICS[camera_name] if streams[topic]]
    if len(present) != 1:
        found = [topic for topic, rows in present if rows]
        raise ValueError(f"{camera_name}: expected exactly one raw/compressed stream, found {found}")
    return present[0]


def decode_image(message: object) -> np.ndarray:
    """Decode sensor_msgs/Image or sensor_msgs/CompressedImage to RGB uint8."""
    data = bytes(message.data)
    if hasattr(message, "format") and not hasattr(message, "encoding"):
        with Image.open(io.BytesIO(data)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)

    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    encoding = str(message.encoding).lower()
    rows = np.frombuffer(data, dtype=np.uint8).reshape(height, step)
    if encoding in {"rgb8", "bgr8"}:
        image = rows[:, : width * 3].reshape(height, width, 3)
        return image[..., ::-1].copy() if encoding == "bgr8" else image.copy()
    if encoding in {"rgba8", "bgra8"}:
        image = rows[:, : width * 4].reshape(height, width, 4)[..., :3]
        return image[..., ::-1].copy() if encoding == "bgra8" else image.copy()
    if encoding in {"mono8", "8uc1"}:
        image = rows[:, :width].reshape(height, width)
        return np.repeat(image[..., None], 3, axis=-1)
    raise ValueError(f"unsupported image encoding {message.encoding!r}")


def measured_state_stream(rows: list[tuple[float, object]]) -> tuple[np.ndarray, np.ndarray]:
    timestamps = []
    states = []
    for timestamp, message in rows:
        positions = dict(zip(message.name, map(float, message.position)))
        missing = [name for name in JOINT_NAMES if name not in positions]
        if missing:
            continue
        timestamps.append(timestamp)
        states.append([positions[name] for name in JOINT_NAMES])
    if not states:
        raise ValueError(f"/joint_states never contained the complete required layout: {JOINT_NAMES}")
    timestamps_array = np.asarray(timestamps, dtype=np.float64)
    states_array = np.asarray(states, dtype=np.float64)
    order = np.argsort(timestamps_array)
    return timestamps_array[order], states_array[order]


def interpolate_states(
    timestamps: np.ndarray,
    states: np.ndarray,
    grid: np.ndarray,
    max_gap_s: float,
) -> np.ndarray:
    gaps = np.diff(timestamps)
    if len(gaps) and float(gaps.max()) > max_gap_s:
        raise ValueError(f"measured joint stream has a {gaps.max():.3f}s gap (limit {max_gap_s:.3f}s)")
    if grid[0] < timestamps[0] or grid[-1] > timestamps[-1]:
        raise ValueError("state interpolation would require extrapolation")
    result = np.column_stack(
        [np.interp(grid, timestamps, states[:, dim]) for dim in range(states.shape[1])]
    ).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("interpolated measured state contains NaN or infinity")
    return result


def write_video(
    path: Path,
    rows: list[tuple[float, object]],
    grid: np.ndarray,
    image_size: int,
    fps: int,
    max_gap_s: float,
) -> dict[str, float | int]:
    valid = []
    corrupt = 0
    for timestamp, message in rows:
        try:
            valid.append((timestamp, decode_image(message)))
        except Exception:
            corrupt += 1
    if not valid:
        raise ValueError(f"{path.name}: no decodable frames")
    timestamps = np.asarray([row[0] for row in valid])
    gaps = np.diff(timestamps)
    metrics = {
        "messages": len(rows),
        "valid": len(valid),
        "corrupt": corrupt,
        "max_valid_gap_s": float(gaps.max()) if len(gaps) else 0.0,
        "leading_gap_s": float(max(0.0, timestamps[0] - grid[0])),
        "trailing_gap_s": float(max(0.0, grid[-1] - timestamps[-1])),
    }
    if (
        corrupt
        or metrics["max_valid_gap_s"] > max_gap_s
        or metrics["leading_gap_s"] > max_gap_s
        or metrics["trailing_gap_s"] > max_gap_s
    ):
        raise ValueError(f"{path.name}: camera continuity failed: {metrics}")

    indices = nearest_indices(timestamps, grid)
    writer = imageio_ffmpeg.write_frames(
        str(path),
        (image_size, image_size),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-crf", "23", "-movflags", "+faststart"],
        macro_block_size=1,
    )
    writer.send(None)
    try:
        for index in indices:
            image = Image.fromarray(valid[int(index)][1]).resize((image_size, image_size), Image.BICUBIC)
            writer.send(np.ascontiguousarray(image, dtype=np.uint8))
    finally:
        writer.close()
    return metrics


def convert_episode(
    raw_path: Path,
    output_dir: Path,
    episode_index: int,
    image_size: int,
    fps: int,
    task: str,
    max_state_gap_s: float,
    max_camera_gap_s: float,
) -> dict:
    streams = read_mcap(raw_path)
    if not streams["/joint_states"]:
        raise ValueError(f"{raw_path}: missing /joint_states")
    cameras = {name: select_camera_stream(streams, name) for name in CAMERA_TOPICS}
    state_timestamps, raw_states = measured_state_stream(streams["/joint_states"])
    base_rows = cameras["cam_high"][1]
    start = max(base_rows[0][0], state_timestamps[0])
    stop = min(base_rows[-1][0], state_timestamps[-1])
    grid = np.arange(start, stop, 1.0 / fps, dtype=np.float64)
    if len(grid) < 51:
        raise ValueError(f"{raw_path}: only {len(grid)} synchronized frames; need at least 51")
    states = interpolate_states(state_timestamps, raw_states, grid, max_state_gap_s)

    stem = f"episode_{episode_index:03d}"
    video_paths = {}
    camera_metrics = {}
    for camera_name, (_, rows) in cameras.items():
        filename = f"{stem}_{camera_name}.mp4"
        video_paths[camera_name] = filename
        camera_metrics[camera_name] = write_video(
            output_dir / filename,
            rows,
            grid,
            image_size,
            fps,
            max_camera_gap_s,
        )

    with h5py.File(output_dir / f"{stem}.hdf5", "w") as handle:
        handle.attrs.update(
            sim=False,
            success=True,
            task_description=task,
            fps=fps,
            source_mcap=str(raw_path),
            action_contract=CONTRACT_NAME,
            action_layout=ACTION_LAYOUT,
            action_storage="measured_state_source_identical_to_observations_qpos",
            first_future_offset=1,
            gripper_mode="absolute_measured_joint_position_rad",
        )
        handle.attrs["camera_resampling_metrics"] = json.dumps(camera_metrics)
        observations = handle.create_group("observations")
        observations.create_dataset("qpos", data=states)
        observations.create_dataset("qvel", data=np.gradient(states, 1.0 / fps, axis=0))
        observations.create_dataset("effort", data=np.zeros_like(states))
        video_group = observations.create_group("video_paths")
        for name, filename in video_paths.items():
            video_group.create_dataset(name, data=filename.encode())
        # This is a measured-state source for sample-time chunk construction,
        # not an action command or a precomputed delta.
        handle.create_dataset("action", data=states)
        handle.create_dataset("relative_action", data=states)
        handle.create_dataset("action_dim_mask", data=np.ones(ACTION_DIM, dtype=np.float32))

    return {
        "episode": episode_index,
        "source": str(raw_path),
        "frames": len(grid),
        "duration_s": float(len(grid) / fps),
        "camera_resampling": camera_metrics,
    }


def load_sources(raw_root: Path, mapping_path: Path | None) -> list[Path]:
    if mapping_path is None:
        sources = sorted(raw_root.rglob("*.mcap"))
    else:
        mapping = json.loads(mapping_path.read_text())
        sources = []
        for item in mapping:
            relative = item.get("path") or item.get("mcap") or item.get("name")
            if relative is None:
                raise ValueError(f"mapping entry has no path/mcap/name: {item}")
            candidate = raw_root / relative
            if candidate.suffix != ".mcap":
                matches = sorted(candidate.rglob("*.mcap")) if candidate.is_dir() else sorted(raw_root.rglob(f"{relative}*.mcap"))
                if len(matches) != 1:
                    raise ValueError(f"mapping entry {relative!r} resolved to {len(matches)} MCAPs")
                candidate = matches[0]
            sources.append(candidate)
    if not sources:
        raise ValueError(f"no MCAP files found under {raw_root}")
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing mapped MCAP files: {missing}")
    return sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-state-gap-s", type=float, default=0.1)
    parser.add_argument("--max-camera-gap-s", type=float, default=0.5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.out.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.out} exists; pass --overwrite to replace it")
        shutil.rmtree(args.out)
    train_dir = args.out / "train"
    train_dir.mkdir(parents=True)
    sources = load_sources(args.raw_root, args.mapping)
    if args.limit is not None:
        sources = sources[: args.limit]

    episodes = []
    for episode_index, source in enumerate(sources):
        print(f"[{episode_index + 1}/{len(sources)}] {source}", flush=True)
        episodes.append(
            convert_episode(
                source,
                train_dir,
                episode_index,
                args.image_size,
                args.fps,
                args.task,
                args.max_state_gap_s,
                args.max_camera_gap_s,
            )
        )
    manifest = {
        "contract": CONTRACT_NAME,
        "task": args.task,
        "fps": args.fps,
        "chunk_size": 50,
        "action_dim": ACTION_DIM,
        "proprio_dim": ACTION_DIM,
        "episodes": episodes,
        "total_frames": sum(item["frames"] for item in episodes),
    }
    (args.out / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"episodes": len(episodes), "total_frames": manifest["total_frames"]}, indent=2))


if __name__ == "__main__":
    main()

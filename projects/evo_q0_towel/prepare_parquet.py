#!/usr/bin/env python3
"""Convert selected LeRobot-v3 Parquet episodes to EVO q0 Cosmos data.

The recorded ``action`` column is intentionally never read. Targets are built
at sample time from the measured ``observation.state`` trajectory.
"""

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

from cosmos_policy.datasets.evo_q0_actions import ACTION_DIM, ACTION_LAYOUT, CONTRACT_NAME


FPS = 30
TASK = "fold the blue towel twice"
IMAGE_COLUMNS = {
    "cam_high": "observation.images.head",
    "cam_left_wrist": "observation.images.left_wrist",
    "cam_right_wrist": "observation.images.right_wrist",
}
REQUIRED_COLUMNS = [
    "observation.state",
    *IMAGE_COLUMNS.values(),
    "timestamp",
    "frame_index",
    "episode_index",
]


def load_selection(path: Path, roots: dict[str, Path]) -> list[tuple[str, Path, int]]:
    """Load an explicit list of accepted episodes; implicit globbing is forbidden."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or not payload:
        raise ValueError("selection must be a non-empty JSON list")
    selected = []
    seen = set()
    for row in payload:
        dataset = str(row["dataset"])
        file_index = int(row["file_index"])
        if dataset not in roots:
            raise ValueError(f"selection references unknown dataset {dataset!r}")
        key = (dataset, file_index)
        if key in seen:
            raise ValueError(f"duplicate selected episode: {key}")
        seen.add(key)
        source = roots[dataset] / "data" / "chunk-000" / f"file-{file_index:03d}.parquet"
        if not source.is_file():
            raise FileNotFoundError(source)
        selected.append((dataset, source, file_index))
    return selected


def decode_frame(cell: dict, column: str, row_index: int, image_size: int) -> np.ndarray:
    encoded = cell.get("bytes")
    if not encoded:
        raise ValueError(f"{column} row {row_index}: missing embedded image bytes")
    try:
        with Image.open(io.BytesIO(encoded)) as image:
            image.load()
            rgb = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
            return np.asarray(rgb, dtype=np.uint8)
    except Exception as error:
        raise ValueError(f"{column} row {row_index}: image decode failed") from error


def open_video_writers(output_dir: Path, stem: str, image_size: int) -> tuple[dict, dict[str, str]]:
    writers = {}
    filenames = {}
    for camera in IMAGE_COLUMNS:
        filename = f"{stem}_{camera}.mp4"
        writer = imageio_ffmpeg.write_frames(
            str(output_dir / filename),
            (image_size, image_size),
            fps=FPS,
            codec="libx264",
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            output_params=["-crf", "23", "-movflags", "+faststart"],
            macro_block_size=1,
        )
        writer.send(None)
        writers[camera] = writer
        filenames[camera] = filename
    return writers, filenames


def convert_episode(
    source: Path,
    output_dir: Path,
    output_index: int,
    dataset: str,
    source_index: int,
    split: str,
    image_size: int,
) -> dict:
    parquet = pq.ParquetFile(source)
    missing = sorted(set(REQUIRED_COLUMNS) - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"{source}: missing columns {missing}")
    if parquet.metadata.num_rows < 51:
        raise ValueError(f"{source}: only {parquet.metadata.num_rows} frames; need at least 51")

    stem = f"episode_{output_index:03d}"
    writers, video_paths = open_video_writers(output_dir, stem, image_size)
    states = []
    timestamps = []
    frame_indices = []
    episode_indices = set()
    row_index = 0
    try:
        for batch in parquet.iter_batches(batch_size=32, columns=REQUIRED_COLUMNS):
            columns = {name: batch.column(name) for name in REQUIRED_COLUMNS}
            for local_index in range(batch.num_rows):
                state = np.asarray(columns["observation.state"][local_index].as_py(), dtype=np.float32)
                if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
                    raise ValueError(f"{source} row {row_index}: invalid measured state {state.shape}")
                states.append(state)
                timestamps.append(float(columns["timestamp"][local_index].as_py()))
                frame_indices.append(int(columns["frame_index"][local_index].as_py()))
                episode_indices.add(int(columns["episode_index"][local_index].as_py()))
                for camera, column in IMAGE_COLUMNS.items():
                    frame = decode_frame(columns[column][local_index].as_py(), column, row_index, image_size)
                    writers[camera].send(np.ascontiguousarray(frame))
                row_index += 1
    finally:
        for writer in writers.values():
            writer.close()

    states_array = np.stack(states)
    timestamps_array = np.asarray(timestamps, dtype=np.float64)
    if frame_indices != list(range(len(frame_indices))):
        raise ValueError(f"{source}: frame_index is not contiguous from zero")
    if len(episode_indices) != 1:
        raise ValueError(f"{source}: contains multiple episode indices {episode_indices}")
    expected = np.arange(len(timestamps_array), dtype=np.float64) / FPS
    max_timing_error = float(np.max(np.abs(timestamps_array - expected)))
    if max_timing_error > 1e-3:
        raise ValueError(f"{source}: timestamps are not on a 30 Hz grid; max error={max_timing_error}")

    with h5py.File(output_dir / f"{stem}.hdf5", "w") as handle:
        handle.attrs.update(
            sim=False,
            success=True,
            task_description=TASK,
            fps=FPS,
            source_parquet=str(source),
            source_dataset=dataset,
            source_file_index=source_index,
            split=split,
            action_contract=CONTRACT_NAME,
            action_layout=ACTION_LAYOUT,
            action_storage="measured_state_source_identical_to_observations_qpos",
            recorded_action_column_used=False,
            first_future_offset=1,
            gripper_mode="absolute_measured_joint_position_rad",
        )
        observations = handle.create_group("observations")
        observations.create_dataset("qpos", data=states_array)
        observations.create_dataset("qvel", data=np.gradient(states_array, 1.0 / FPS, axis=0))
        observations.create_dataset("effort", data=np.zeros_like(states_array))
        video_group = observations.create_group("video_paths")
        for name, filename in video_paths.items():
            video_group.create_dataset(name, data=filename.encode())
        handle.create_dataset("action", data=states_array)
        handle.create_dataset("relative_action", data=states_array)
        handle.create_dataset("action_dim_mask", data=np.ones(ACTION_DIM, dtype=np.float32))

    return {
        "episode": output_index,
        "split": split,
        "source_dataset": dataset,
        "source_file_index": source_index,
        "source": str(source),
        "frames": len(states_array),
        "duration_s": len(states_array) / FPS,
        "max_timing_error_s": max_timing_error,
    }


def yam_validation_indices(count: int, val_count: int) -> set[int]:
    if val_count < 0 or val_count >= count:
        raise ValueError(f"val_count must be in [0, {count - 1}], got {val_count}")
    if val_count == 0:
        return set()
    return set(np.linspace(0, count - 1, val_count + 2, dtype=int)[1:-1].tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True, help="NAME=/local/dataset/root")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--val-count", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    roots = {}
    for value in args.dataset:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"invalid --dataset {value!r}; expected NAME=/path")
        roots[name] = Path(raw_path)
    selected = load_selection(args.selection, roots)
    if args.out.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.out} exists; pass --overwrite to replace it")
        shutil.rmtree(args.out)
    (args.out / "train").mkdir(parents=True)
    (args.out / "val").mkdir(parents=True)

    val_indices = yam_validation_indices(len(selected), args.val_count)
    episodes = []
    for output_index, (dataset, source, source_index) in enumerate(selected):
        split = "val" if output_index in val_indices else "train"
        print(f"[{output_index + 1}/{len(selected)}] {split} {dataset}/file-{source_index:03d}", flush=True)
        episodes.append(
            convert_episode(
                source,
                args.out / split,
                output_index,
                dataset,
                source_index,
                split,
                args.image_size,
            )
        )

    manifest = {
        "contract": CONTRACT_NAME,
        "task": TASK,
        "fps": FPS,
        "chunk_size": 50,
        "action_dim": ACTION_DIM,
        "proprio_dim": ACTION_DIM,
        "selection_file": str(args.selection),
        "validation_method": "yam_linspace_evenly_spaced",
        "validation_output_indices": sorted(val_indices),
        "episodes": episodes,
        "counts": {
            "total": len(episodes),
            "train": len(episodes) - len(val_indices),
            "val": len(val_indices),
        },
        "total_frames": sum(row["frames"] for row in episodes),
    }
    (args.out / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(args.selection, args.out / "accepted_selection.json")
    print(json.dumps({"counts": manifest["counts"], "frames": manifest["total_frames"]}, indent=2))


if __name__ == "__main__":
    main()

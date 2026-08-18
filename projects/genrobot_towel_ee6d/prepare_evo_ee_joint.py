#!/usr/bin/env python3
"""Convert accepted Evo LeRobot Parquets directly to shared-35 Cosmos data.

Targets come only from measured ``observation.state``. The recorded command
column is deliberately ignored. Each output action row is constructed later by
the dataset loader from absolute base-frame palm FK and measured joint state.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
from pathlib import Path

import h5py
import imageio_ffmpeg
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from cosmos_policy.datasets.ee_q0_actions import (
    CONTRACT_NAME,
    EVO_STORAGE_CONTRACT,
    EVO_TO_UMI_ORIENTATION_OFFSET_WXYZ,
    NORMALIZATION_CONTRACT,
    SHARED_ACTION_DIM,
    canonical_shared_statistics,
)
from projects.genrobot_towel_ee6d.convert_evo_fk import FK, convert_states
from projects.evo_q0_towel.newoffice73_split import (
    VALIDATION_OUTPUT_INDICES,
    load_original_split,
    split_summary,
)


FPS = 30
TASK = "fold the blue towel twice"
PROPRIO_DIM = 17
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
    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or not payload:
        raise ValueError("selection must be a non-empty JSON list")
    selected, seen = [], set()
    for row in payload:
        dataset, file_index = str(row["dataset"]), int(row["file_index"])
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


def open_video_writers(output_dir: Path, stem: str, image_size: int):
    writers, filenames = {}, {}
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
        writers[camera], filenames[camera] = writer, filename
    return writers, filenames


def validation_indices(count: int, val_count: int) -> set[int]:
    if not 0 <= val_count < count:
        raise ValueError(f"val_count must be in [0,{count - 1}]")
    if val_count == 0:
        return set()
    return set(np.linspace(0, count - 1, val_count + 2, dtype=int)[1:-1].tolist())


def convert_episode(
    source_path: Path,
    output_dir: Path,
    output_index: int,
    source_dataset: str,
    source_index: int,
    split: str,
    image_size: int,
    fk: FK,
) -> dict:
    parquet = pq.ParquetFile(source_path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"{source_path}: missing columns {missing}")
    if parquet.metadata.num_rows < 51:
        raise ValueError(f"{source_path}: only {parquet.metadata.num_rows} frames; need at least 51")

    stem = f"episode_{output_index:03d}"
    writers, video_paths = open_video_writers(output_dir, stem, image_size)
    states, timestamps, frame_indices, episode_indices = [], [], [], set()
    row_index = 0
    try:
        for batch in parquet.iter_batches(batch_size=32, columns=REQUIRED_COLUMNS):
            columns = {name: batch.column(name) for name in REQUIRED_COLUMNS}
            for local_index in range(batch.num_rows):
                state = np.asarray(columns["observation.state"][local_index].as_py(), dtype=np.float32)
                if state.shape != (PROPRIO_DIM,) or not np.isfinite(state).all():
                    raise ValueError(f"{source_path} row {row_index}: invalid state {state.shape}")
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

    measured_states = np.stack(states)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if frame_indices != list(range(len(frame_indices))):
        raise ValueError(f"{source_path}: frame_index is not contiguous from zero")
    if len(episode_indices) != 1:
        raise ValueError(f"{source_path}: contains multiple episode indices {episode_indices}")
    expected_time = np.arange(len(timestamps), dtype=np.float64) / FPS
    timing_error = float(np.max(np.abs(timestamps - expected_time)))
    if timing_error > 1e-3:
        raise ValueError(f"{source_path}: not a 30 Hz grid; max error={timing_error}")

    absolute_source, proprio = convert_states(measured_states, fk)
    if absolute_source.shape != (len(measured_states), SHARED_ACTION_DIM):
        raise RuntimeError("FK conversion returned the wrong shared action shape")

    hdf5_path = output_dir / f"{stem}.hdf5"
    with h5py.File(hdf5_path, "w") as handle:
        handle.attrs.update(
            sim=False,
            success=True,
            task_description=TASK,
            fps=FPS,
            source_parquet=str(source_path),
            source_dataset=source_dataset,
            source_file_index=source_index,
            split=split,
            action_contract=CONTRACT_NAME,
            action_storage_contract=EVO_STORAGE_CONTRACT,
            normalization_contract=NORMALIZATION_CONTRACT,
            source_pose_link="base_link",
            fk_root="base_link",
            fk_tips="left_palm,right_palm",
            position_post_transform="none",
            orientation_post_transform="right_multiply_q_offset_wxyz",
            orientation_offset_wxyz="0,0.7071067811865476,0,0.7071067811865476",
            delta_reference_frame="query_body",
            recorded_action_column_used=False,
            first_future_offset=1,
            gripper_mode="absolute_canonical_aperture_closed0_open1",
        )
        observations = handle.create_group("observations")
        observations.create_dataset("qpos", data=proprio, compression="gzip")
        observations.create_dataset("qvel", data=np.gradient(proprio, 1.0 / FPS, axis=0), compression="gzip")
        observations.create_dataset("effort", data=np.zeros_like(proprio), compression="gzip")
        video_group = observations.create_group("video_paths")
        for camera, filename in video_paths.items():
            video_group.create_dataset(camera, data=filename.encode())
        handle.create_dataset("action", data=absolute_source, compression="gzip")
        handle.create_dataset("relative_action", data=absolute_source, compression="gzip")
        handle.create_dataset("action_dim_mask", data=np.ones(SHARED_ACTION_DIM, dtype=np.float32))
        handle.create_dataset("proprio_dim_mask", data=np.ones(PROPRIO_DIM, dtype=np.float32))

    return {
        "output_index": output_index,
        "split": split,
        "source_dataset": source_dataset,
        "source_file_index": source_index,
        "frames": len(measured_states),
        "duration_s": len(measured_states) / FPS,
        "max_timing_error_s": timing_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True, help="NAME=/dataset/root")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--val-count", type=int, default=6)
    parser.add_argument("--expected-episodes", type=int, default=73)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    roots = {}
    if args.val_count != 6 or args.expected_episodes != 73:
        raise ValueError(
            "the original Evo-only ablation split requires --val-count 6 and --expected-episodes 73"
        )
    for value in args.dataset:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"invalid --dataset {value!r}; expected NAME=/path")
        roots[name] = Path(raw_path)
    split_rows = load_original_split(args.selection)
    selected = load_selection(args.selection, roots)
    selected_identities = [(dataset, file_index) for dataset, _, file_index in selected]
    contract_identities = [(str(row["dataset"]), int(row["file_index"])) for row in split_rows]
    if selected_identities != contract_identities:
        raise RuntimeError("resolved episode ordering differs from the original Evo-only split contract")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing output {args.out}")
    (args.out / "train").mkdir(parents=True)
    (args.out / "val").mkdir(parents=True)

    validation = set(VALIDATION_OUTPUT_INDICES)
    fk = FK(args.urdf)
    episodes = []
    for output_index, (dataset, source, source_index) in enumerate(selected):
        split = "val" if output_index in validation else "train"
        print(f"[{output_index + 1}/{len(selected)}] {split} {dataset}/file-{source_index:03d}", flush=True)
        episodes.append(
            convert_episode(source, args.out / split, output_index, dataset, source_index, split, args.image_size, fk)
        )

    statistics = {key: value.tolist() for key, value in canonical_shared_statistics().items()}
    (args.out / "dataset_statistics.json").write_text(json.dumps(statistics, indent=2) + "\n")
    shutil.copy2(args.selection, args.out / "accepted_selection.json")
    manifest = {
        "action_contract": CONTRACT_NAME,
        "storage_contract": EVO_STORAGE_CONTRACT,
        "normalization_contract": NORMALIZATION_CONTRACT,
        "action_dim": SHARED_ACTION_DIM,
        "proprio_dim": PROPRIO_DIM,
        "chunk_size": 50,
        "fps": FPS,
        "task": TASK,
        "recorded_action_column_used": False,
        "first_future_offset": 1,
        "fk": {
            "root": "base_link",
            "tips": ["left_palm", "right_palm"],
            "position_post_transform": "none",
            "orientation_post_transform": "right_multiply_q_offset_wxyz",
            "orientation_offset_wxyz": EVO_TO_UMI_ORIENTATION_OFFSET_WXYZ.tolist(),
            "includes_joint": "carriage_joint",
            "urdf_sha256": hashlib.sha256(args.urdf.read_bytes()).hexdigest(),
        },
        "validation_method": "yam_linspace_evenly_spaced",
        "validation_output_indices": sorted(validation),
        "ablation_split": split_summary(split_rows),
        "counts": {"total": len(episodes), "train": len(episodes) - len(validation), "val": len(validation)},
        "total_frames": sum(row["frames"] for row in episodes),
        "episodes": episodes,
    }
    (args.out / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"counts": manifest["counts"], "total_frames": manifest["total_frames"]}, indent=2))


if __name__ == "__main__":
    main()

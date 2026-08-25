#!/usr/bin/env python3
"""Convert LeHome top-short demonstrations into a q0-ready Zarr store.

The source format is LeRobot v3: Parquet stores timestamped state rows and
shared AV1 MP4 chunks store RGB frames.  This script preserves the episode
boundaries from ``meta/episodes`` and does not use the source ``action``
column.  It deliberately materializes RGB to 320x240 Zarr once, so training
workers never seek AV1 video during a forward pass.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import av
import cv2
import numpy as np
import pyarrow.parquet as pq
import zarr
from numcodecs import Blosc


TASK = "record_top_short_release_10"
CAMERAS = {
    "top_rgb": "observation.images.top_rgb",
    "left_rgb": "observation.images.left_rgb",
    "right_rgb": "observation.images.right_rgb",
}


def parquet_rows(root: Path, relative: str):
    files = sorted(root.glob(relative))
    if not files:
        raise FileNotFoundError(f"no files matching {relative} under {root}")
    return [pq.read_table(path) for path in files]


def session_state(session: Path) -> np.ndarray:
    tables = parquet_rows(session, "data/**/*.parquet")
    state = np.concatenate(
        [np.asarray(table["observation.state"].to_pylist(), dtype=np.float32) for table in tables], axis=0
    )
    if state.ndim != 2 or state.shape[1] != 12:
        raise ValueError(f"expected [N,12] observation.state in {session}, got {state.shape}")
    return state


def session_episodes(session: Path) -> list[dict]:
    tables = parquet_rows(session, "meta/episodes/**/*.parquet")
    rows = []
    for table in tables:
        rows.extend(table.select(["episode_index", "length"]).to_pylist())
    rows.sort(key=lambda row: row["episode_index"])
    if len(rows) != 25:
        raise ValueError(f"expected exactly 25 episodes in {session}, found {len(rows)}")
    return rows


def decode_video_into(video_path: Path, dst, offset: int, expected_frames: int) -> None:
    """Decode one CFR LeHome stream to a chunked Zarr array without large RAM use."""
    chunk_size = 64
    frames = []
    written = 0
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if written >= expected_frames:
                break
            rgb = frame.to_ndarray(format="rgb24")
            frames.append(cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_AREA))
            written += 1
            if len(frames) == chunk_size:
                start = offset + written - len(frames)
                dst[start : start + len(frames)] = np.asarray(frames, dtype=np.uint8)
                frames.clear()
    if frames:
        start = offset + written - len(frames)
        dst[start : start + len(frames)] = np.asarray(frames, dtype=np.uint8)
    if written != expected_frames:
        raise RuntimeError(
            f"{video_path} decoded {written} frames; expected exactly {expected_frames}. "
            "Do not train from a partially decoded store."
        )


def make_manifest(episode_ids: list[str], seed: int) -> dict:
    if len(episode_ids) != 250 or len(set(episode_ids)) != 250:
        raise ValueError("expected exactly 250 unique top-short episodes")
    permutation = np.random.default_rng(seed).permutation(len(episode_ids))
    val_indices = set(permutation[:20].tolist())
    return {
        "task": TASK,
        "seed": seed,
        "contract": "q0_observed_state_delta_v1",
        "train": [episode_id for i, episode_id in enumerate(episode_ids) if i not in val_indices],
        "val": [episode_id for i, episode_id in enumerate(episode_ids) if i in val_indices],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True, help="directory containing record_top_short_release_10")
    parser.add_argument("--out", type=Path, required=True, help="new Zarr output path")
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    task_root = args.raw_root / TASK
    sessions = sorted(path for path in task_root.iterdir() if path.is_dir() and path.name.isdigit())
    if [path.name for path in sessions] != [f"{i:03d}" for i in range(1, 11)]:
        raise ValueError(f"expected sessions 001..010 under {task_root}, found {[p.name for p in sessions]}")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.out}")

    session_payload = []
    total_frames = 0
    episode_ids = []
    episode_ends = []
    for session in sessions:
        state = session_state(session)
        episodes = session_episodes(session)
        if sum(row["length"] for row in episodes) != len(state):
            raise ValueError(f"episode lengths do not match state rows for {session}")
        session_payload.append((session, state, episodes))
        total_frames += len(state)
        for row in episodes:
            episode_ids.append(f"{session.name}/{int(row['episode_index']):03d}")
            episode_ends.append((episode_ends[-1] if episode_ends else 0) + int(row["length"]))

    manifest = make_manifest(episode_ids, args.seed)
    temporary_out = args.out.with_name(args.out.name + ".incomplete")
    if temporary_out.exists():
        raise FileExistsError(f"remove stale incomplete output manually: {temporary_out}")
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    store = zarr.DirectoryStore(str(temporary_out))
    root = zarr.group(store=store, overwrite=True)
    root.create_dataset("state", shape=(total_frames, 12), chunks=(1024, 12), dtype="f4", compressor=compressor)
    root.create_dataset("meta/episode_ends", data=np.asarray(episode_ends, dtype=np.int64), compressor=compressor)
    for camera in CAMERAS:
        root.create_dataset(
            f"images/{camera}",
            shape=(total_frames, 240, 320, 3),
            chunks=(16, 240, 320, 3),
            dtype="u1",
            compressor=compressor,
        )
    root.attrs.update({"fps": 30, "episode_ids": episode_ids, "source_task": TASK, "source_action_used": False})

    offset = 0
    for session, state, _episodes in session_payload:
        print(f"processing {session.name}: {len(state)} frames", flush=True)
        root["state"][offset : offset + len(state)] = state
        for output_name, source_name in CAMERAS.items():
            videos = sorted(session.glob(f"videos/{source_name}/**/*.mp4"))
            if len(videos) != 1:
                raise FileNotFoundError(f"expected one {source_name} MP4 in {session}, found {videos}")
            decode_video_into(videos[0], root[f"images/{output_name}"], offset, len(state))
        offset += len(state)

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary_out.rename(args.out)
    print(f"wrote {args.out} ({total_frames} frames), {args.manifest_out}")
    print(f"split: {len(manifest['train'])} train, {len(manifest['val'])} val")


if __name__ == "__main__":
    main()

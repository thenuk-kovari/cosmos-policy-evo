#!/usr/bin/env python3
"""Recompute Evo q0 statistics from the exact initial train episode split.

Only ``observation.state`` is read.  With S3 Parquet inputs this uses column
and range reads, so embedded camera data is not downloaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from cosmos_policy.datasets.evo_q0_actions import ACTION_DIM, CHUNK_SIZE, calculate_q0_action_statistics
from projects.evo_q0_towel.newoffice73_split import load_original_split, split_summary


STATISTIC_NAMES = ("min", "max", "mean", "std", "median")


def vector_statistics(values: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_min": values.min(axis=0).astype(np.float32),
        f"{prefix}_max": values.max(axis=0).astype(np.float32),
        f"{prefix}_mean": values.mean(axis=0).astype(np.float32),
        f"{prefix}_std": values.std(axis=0).astype(np.float32),
        f"{prefix}_median": np.median(values, axis=0).astype(np.float32),
    }


def parse_roots(values: list[str]) -> dict[str, str]:
    roots: dict[str, str] = {}
    for value in values:
        name, separator, root = value.partition("=")
        if not separator or not name or not root:
            raise ValueError(f"invalid --dataset {value!r}; expected NAME=/path or NAME=s3://bucket/prefix")
        roots[name] = root.rstrip("/")
    return roots


def read_state(root: str, file_index: int, s3: pafs.S3FileSystem | None) -> np.ndarray:
    suffix = f"data/chunk-000/file-{file_index:03d}.parquet"
    if root.startswith("s3://"):
        if s3 is None:
            raise RuntimeError("S3 filesystem was not initialized")
        parsed = urlparse(root)
        source = f"{parsed.netloc}/{parsed.path.strip('/')}/{suffix}"
        table = pq.read_table(source, filesystem=s3, columns=["observation.state"])
    else:
        source = str(Path(root) / suffix)
        table = pq.read_table(source, columns=["observation.state"])
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != ACTION_DIM or not np.isfinite(states).all():
        raise ValueError(f"{source}: expected finite [T,{ACTION_DIM}] observation.state, got {states.shape}")
    return states


def comparison(recomputed: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> dict[str, object]:
    details: dict[str, object] = {}
    exact = True
    for key in sorted(recomputed):
        actual = np.asarray(recomputed[key], dtype=np.float32)
        expected = np.asarray(reference[key], dtype=np.float32)
        if actual.shape != expected.shape:
            details[key] = {"shape_match": False, "actual": list(actual.shape), "reference": list(expected.shape)}
            exact = False
            continue
        difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        key_exact = bool(np.array_equal(actual, expected))
        exact &= key_exact
        details[key] = {
            "shape_match": True,
            "float32_exact": key_exact,
            "max_abs_difference": float(difference.max(initial=0.0)),
            "different_dimensions": np.flatnonzero(difference != 0).tolist(),
        }
    return {"all_float32_exact": exact, "statistics": details}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True, help="NAME=/root or NAME=s3://bucket/prefix")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--reference-statistics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--region", default="us-west-2")
    args = parser.parse_args()

    roots = parse_roots(args.dataset)
    split_rows = load_original_split(args.selection)
    missing_roots = sorted({str(row["dataset"]) for row in split_rows} - set(roots))
    if missing_roots:
        raise ValueError(f"missing dataset roots: {missing_roots}")
    s3 = pafs.S3FileSystem(region=args.region) if any(root.startswith("s3://") for root in roots.values()) else None

    trajectories: list[np.ndarray] = []
    episodes: list[dict[str, object]] = []
    for row in split_rows:
        if row["split"] != "train":
            continue
        dataset, file_index = str(row["dataset"]), int(row["file_index"])
        print(f"[{len(trajectories) + 1}/67] {dataset}/file-{file_index:03d}", flush=True)
        states = read_state(roots[dataset], file_index, s3)
        trajectories.append(states)
        episodes.append({**row, "frames": len(states)})

    action = calculate_q0_action_statistics(trajectories, chunk_size=CHUNK_SIZE)
    proprio = vector_statistics(np.concatenate(trajectories, axis=0), "proprio")
    recomputed = {**action, **proprio}
    reference = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in json.loads(args.reference_statistics.read_text()).items()
    }
    report = {
        **split_summary(split_rows),
        "train_frames": int(sum(len(states) for states in trajectories)),
        "episodes": episodes,
        "comparison": comparison(recomputed, reference),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "dataset_statistics.json").write_text(
        json.dumps({key: value.tolist() for key, value in recomputed.items()}, indent=2) + "\n"
    )
    (args.out / "comparison_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "episodes"}, indent=2))
    if not report["comparison"]["all_float32_exact"]:
        raise SystemExit("recomputed statistics do not exactly match the original Evo-only artifact")


if __name__ == "__main__":
    main()

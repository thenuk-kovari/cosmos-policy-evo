"""Immutable episode split used by the original Evo-only towel run.

This is an ablation identity contract, not a split-generation policy.  The
accepted episode list, its ordering, and the six validation identities must
remain byte-for-byte/identity-for-identity compatible with the initial run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


SELECTION_SHA256 = "dad1b8b602d4458a48f1af3a6a1e37870a3c800fb6803b74f9e36ebaf750cb0e"
TOTAL_EPISODES = 73
VALIDATION_OUTPUT_INDICES = frozenset((10, 20, 30, 41, 51, 61))
VALIDATION_IDENTITIES = frozenset(
    (
        ("new50", 10),
        ("new50", 20),
        ("new50", 30),
        ("new50", 42),
        ("new25", 3),
        ("new25", 15),
    )
)


def load_original_split(selection_path: str | Path) -> list[dict[str, object]]:
    """Load and validate the exact initial 67-train/6-validation split."""
    path = Path(selection_path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SELECTION_SHA256:
        raise ValueError(
            f"{path}: selection SHA-256 {digest} does not match the original "
            f"Evo-only selection {SELECTION_SHA256}"
        )
    payload = json.loads(raw)
    if not isinstance(payload, list) or len(payload) != TOTAL_EPISODES:
        raise ValueError(f"{path}: expected exactly {TOTAL_EPISODES} accepted episodes")

    rows: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for output_index, raw_row in enumerate(payload):
        identity = (str(raw_row["dataset"]), int(raw_row["file_index"]))
        if identity in seen:
            raise ValueError(f"{path}: duplicate episode identity {identity}")
        seen.add(identity)
        rows.append(
            {
                "output_index": output_index,
                "dataset": identity[0],
                "file_index": identity[1],
                "split": "val" if output_index in VALIDATION_OUTPUT_INDICES else "train",
            }
        )

    actual_validation = {
        (str(row["dataset"]), int(row["file_index"]))
        for row in rows
        if row["split"] == "val"
    }
    if actual_validation != VALIDATION_IDENTITIES:
        raise RuntimeError(
            "original selection ordering no longer produces the original validation identities: "
            f"{sorted(actual_validation)} != {sorted(VALIDATION_IDENTITIES)}"
        )
    return rows


def split_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "val"]
    return {
        "contract": "evo_newoffice73_initial_run_split_v1",
        "selection_sha256": SELECTION_SHA256,
        "counts": {"total": len(rows), "train": len(train), "val": len(validation)},
        "validation_output_indices": sorted(VALIDATION_OUTPUT_INDICES),
        "validation_identities": [
            {"dataset": dataset, "file_index": file_index}
            for dataset, file_index in sorted(VALIDATION_IDENTITIES)
        ],
    }

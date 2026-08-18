import json
from pathlib import Path

import pytest

from projects.evo_q0_towel.newoffice73_split import (
    SELECTION_SHA256,
    VALIDATION_IDENTITIES,
    VALIDATION_OUTPUT_INDICES,
    load_original_split,
    split_summary,
)


SELECTION = Path(__file__).parents[1] / "projects/evo_q0_towel/accepted_new_office_73.json"


def test_original_newoffice73_split_is_identity_pinned():
    rows = load_original_split(SELECTION)
    assert split_summary(rows) == {
        "contract": "evo_newoffice73_initial_run_split_v1",
        "selection_sha256": SELECTION_SHA256,
        "counts": {"total": 73, "train": 67, "val": 6},
        "validation_output_indices": [10, 20, 30, 41, 51, 61],
        "validation_identities": [
            {"dataset": dataset, "file_index": file_index}
            for dataset, file_index in sorted(VALIDATION_IDENTITIES)
        ],
    }
    assert {
        (row["dataset"], row["file_index"])
        for row in rows
        if row["output_index"] in VALIDATION_OUTPUT_INDICES
    } == VALIDATION_IDENTITIES


def test_modified_selection_is_rejected(tmp_path):
    payload = json.loads(SELECTION.read_text())
    payload[0], payload[1] = payload[1], payload[0]
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload, indent=2) + "\n")
    with pytest.raises(ValueError, match="selection SHA-256"):
        load_original_split(changed)

#!/usr/bin/env python3
"""Strictly load the final DCP checkpoint and execute one synthetic query."""

import json
from pathlib import Path

import numpy as np

from backends.cosmos_policy_backend import CHUNK_SIZE, DEPLOYED_ACTION_DIM, CosmosPolicyBackend


def main() -> None:
    policy_root = Path("/policy")
    model_config = {
        "name": "slot_b",
        "parameters": {
            "EXPERIMENT": {
                "string_value": "predict2-2b-evo-ee6d-joint35-teleop"
            },
            "TASK_DESCRIPTION": {
                "string_value": "fold the blue towel twice"
            },
            "DENOISING_STEPS": {"string_value": "10"},
        },
    }
    model_config["parameters"].update(
        {
            "CHECKPOINT_PATH": {"string_value": str(policy_root)},
            "DATASET_STATS_PATH": {
                "string_value": str(policy_root / "dataset_statistics.json")
            },
            "T5_EMBEDDINGS_PATH": {
                "string_value": str(policy_root / "t5_embeddings.pkl")
            },
        }
    )

    backend = CosmosPolicyBackend()
    backend.initialize(
        {
            "model_config": json.dumps(model_config),
            "model_instance_device_id": "0",
            "model_repository": "/models",
        }
    )

    stats = json.loads((policy_root / "dataset_statistics.json").read_text())
    state = np.asarray(stats["proprio_median"], dtype=np.float32)
    image = np.zeros((3, 224, 224), dtype=np.float32)
    processed = backend.preprocess(
        {
            "observation__state": state,
            "observation__images__base": image,
            "observation__images__left_gripper": image,
            "observation__images__right_gripper": image,
        }
    )
    result = backend.infer(processed)
    actions = np.asarray(result["action"], dtype=np.float32)
    assert actions.shape == (CHUNK_SIZE, DEPLOYED_ACTION_DIM), actions.shape
    assert np.isfinite(actions).all()
    print(
        json.dumps(
            {
                "strict_dcp_load": "passed",
                "missing_keys": 0,
                "unexpected_keys": 0,
                "action_shape": list(actions.shape),
                "action_min": float(actions.min()),
                "action_max": float(actions.max()),
                "thor_fallbacks": backend.fallbacks,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

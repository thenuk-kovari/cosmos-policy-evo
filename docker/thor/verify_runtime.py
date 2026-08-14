#!/usr/bin/env python3
"""Fast, non-inference checks for the self-contained Thor image."""

import json
import os
import pickle
import platform
import sys
from pathlib import Path

import numpy as np
import torch

torch.compile = lambda fn=None, *args, **kwargs: fn if fn is not None else (lambda f: f)

if not any("genrobot" in arg.lower() or "ee6d" in arg.lower() for arg in sys.argv):
    sys.argv.append("genrobot-ee6d")

from cosmos_policy.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM
from cosmos_policy.experiments.robot.cosmos_utils import get_action, get_model


def main() -> None:
    assert platform.machine() == "aarch64", platform.machine()
    assert (ACTION_DIM, PROPRIO_DIM, NUM_ACTIONS_CHUNK) == (35, 17, 50)
    assert torch.version.cuda is not None
    assert callable(get_model) and callable(get_action)

    config = json.loads(Path("/opt/cosmos-policy/deploy/triton/policy_config.json").read_text())
    assert config["input_features"]["observation.state"]["shape"] == [17]
    assert config["output_features"]["action"]["shape"] == [17]
    assert config["n_action_steps"] == 50

    stats_path = os.environ.get("COSMOS_STATS_PATH")
    if stats_path:
        stats = json.loads(Path(stats_path).read_text())
        assert np.asarray(stats["actions_min"]).shape == (35,)
        assert np.asarray(stats["proprio_min"]).shape == (17,)

    t5_path = os.environ.get("COSMOS_T5_PATH")
    if t5_path:
        with Path(t5_path).open("rb") as handle:
            embeddings = pickle.load(handle)
        assert embeddings
        for value in embeddings.values():
            assert tuple(value.shape[-2:]) == (512, 1024)

    print(
        json.dumps(
            {
                "machine": platform.machine(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "action_dim": ACTION_DIM,
                "proprio_dim": PROPRIO_DIM,
                "chunk_size": NUM_ACTIONS_CHUNK,
                "get_model": get_model.__module__,
                "get_action": get_action.__module__,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

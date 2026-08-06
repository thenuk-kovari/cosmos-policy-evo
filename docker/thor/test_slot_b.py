#!/usr/bin/env python3
"""Send one synthetic request through a running slot-B Triton HTTP server."""

import json
from urllib.request import Request, urlopen

import numpy as np


def main() -> None:
    stats_path = "/models/slot_b/checkpoint/dataset_statistics.json"
    with open(stats_path, encoding="utf-8") as handle:
        stats = json.load(handle)

    state = np.asarray(stats["proprio_median"], dtype=np.float32)
    image = np.zeros((3, 224, 224), dtype=np.float32)
    arrays = {
        "observation__state": state,
        "observation__images__base": image,
        "observation__images__left_gripper": image,
        "observation__images__right_gripper": image,
    }

    request_body = {
        "inputs": [
            {
                "name": name,
                "shape": list(array.shape),
                "datatype": "FP32",
                "data": array.reshape(-1).tolist(),
            }
            for name, array in arrays.items()
        ],
        "outputs": [{"name": "action"}],
    }
    request = Request(
        "http://127.0.0.1:8000/v2/models/slot_b/infer",
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=300) as response:
        result = json.load(response)

    output = next(value for value in result["outputs"] if value["name"] == "action")
    action = np.asarray(output["data"], dtype=np.float32).reshape(output["shape"])
    assert action.shape == (50, 29), action.shape
    assert np.isfinite(action).all()
    print(
        {
            "ready": True,
            "shape": action.shape,
            "min": float(action.min()),
            "max": float(action.max()),
        }
    )


if __name__ == "__main__":
    main()

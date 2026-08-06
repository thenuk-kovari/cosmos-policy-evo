#!/usr/bin/env python3
"""Send one synthetic request through a running slot-B Triton gRPC server."""

import json

import numpy as np
import tritonclient.grpc as grpc


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

    inputs = []
    for name, array in arrays.items():
        value = grpc.InferInput(name, array.shape, "FP32")
        value.set_data_from_numpy(array)
        inputs.append(value)

    client = grpc.InferenceServerClient("127.0.0.1:8001")
    action = client.infer("slot_b", inputs=inputs).as_numpy("action")
    assert action.shape == (50, 29), action.shape
    assert np.isfinite(action).all()
    print(
        {
            "ready": client.is_model_ready("slot_b"),
            "shape": action.shape,
            "min": float(action.min()),
            "max": float(action.max()),
        }
    )


if __name__ == "__main__":
    main()

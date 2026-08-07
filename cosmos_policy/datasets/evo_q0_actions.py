"""EVO fixed-anchor future-state action representation.

The stored per-frame action source is the *measured* 17-D robot state, not a
command stream.  A training sample anchored at frame ``i`` is constructed as::

    arm/elevator action[k] = measured_state[i + k + 1] - measured_state[i]
    gripper action[k]      = measured_gripper[i + k + 1]

All 50 rows therefore share one observation-time anchor.  Near the end of an
episode, future indices are clipped to the final measured state.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ACTION_DIM = 17
PROPRIO_DIM = 17
CHUNK_SIZE = 50

LEFT_JOINTS = slice(0, 7)
LEFT_GRIPPER = 7
RIGHT_JOINTS = slice(8, 15)
RIGHT_GRIPPER = 15
ELEVATOR = 16
GRIPPER_DIMS = (LEFT_GRIPPER, RIGHT_GRIPPER)

ACTION_LAYOUT = (
    "left_joint_query_delta7,left_gripper_absolute,right_joint_query_delta7,"
    "right_gripper_absolute,elevator_query_delta"
)
CONTRACT_NAME = "evo_q0_observed_state_left_first_v1"


def validate_state_trajectory(states: np.ndarray) -> np.ndarray:
    """Return ``states`` as float32 after enforcing the 17-D contract."""
    states = np.asarray(states)
    if states.ndim != 2 or states.shape[1] != PROPRIO_DIM:
        raise ValueError(f"expected measured state trajectory [T,{PROPRIO_DIM}], got {states.shape}")
    if len(states) == 0:
        raise ValueError("measured state trajectory is empty")
    if not np.isfinite(states).all():
        raise ValueError("measured state trajectory contains NaN or infinity")
    return states.astype(np.float32, copy=False)


def future_indices(anchor_index: int, chunk_size: int, num_steps: int) -> np.ndarray:
    """Indices ``anchor+1 ... anchor+chunk_size``, terminal-state padded."""
    if not 0 <= anchor_index < num_steps:
        raise IndexError(f"anchor {anchor_index} outside trajectory of length {num_steps}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    return np.minimum(anchor_index + 1 + np.arange(chunk_size), num_steps - 1)


def build_q0_anchored_chunk(
    measured_states: np.ndarray,
    anchor_index: int,
    chunk_size: int = CHUNK_SIZE,
) -> np.ndarray:
    """Build one fixed-q0 action chunk from future *measured* states.

    Arm joints (radians) and elevator (metres) are query-relative. Radial
    grippers remain absolute measured joint positions in radians because their
    state rather than an increment to integrate.
    """
    states = validate_state_trajectory(measured_states)
    indices = future_indices(anchor_index, chunk_size, len(states))
    future = states[indices].copy()
    chunk = future - states[anchor_index]
    chunk[:, GRIPPER_DIMS] = future[:, GRIPPER_DIMS]
    return chunk.astype(np.float32, copy=False)


def normalize_action_chunk(chunk: np.ndarray, statistics: dict[str, np.ndarray]) -> np.ndarray:
    """Apply Cosmos' min/max action normalization to an already-built chunk."""
    chunk = np.asarray(chunk, dtype=np.float32)
    minimum = np.asarray(statistics["actions_min"], dtype=np.float32)
    maximum = np.asarray(statistics["actions_max"], dtype=np.float32)
    if minimum.shape != (ACTION_DIM,) or maximum.shape != (ACTION_DIM,):
        raise ValueError("q0 action statistics must each contain 17 values")
    denominator = np.where(np.abs(maximum - minimum) < 1e-8, 1.0, maximum - minimum)
    return (2.0 * ((chunk - minimum) / denominator) - 1.0).astype(np.float32)


def calculate_q0_action_statistics(
    trajectories: list[np.ndarray],
    chunk_size: int = CHUNK_SIZE,
    max_median_samples: int = 1_000_000,
) -> dict[str, np.ndarray]:
    """Calculate statistics over the chunks the model will actually see.

    Min/max/mean/std are exact. Median is calculated from a deterministic,
    uniformly-strided sample when the complete set exceeds
    ``max_median_samples`` rows.
    """
    states_list = [validate_state_trajectory(x) for x in trajectories]
    total_rows = sum(len(x) * chunk_size for x in states_list)
    if total_rows == 0:
        raise ValueError("cannot calculate statistics for an empty dataset")
    median_stride = max(1, math.ceil(total_rows / max_median_samples))

    minimum = np.full(ACTION_DIM, np.inf, dtype=np.float64)
    maximum = np.full(ACTION_DIM, -np.inf, dtype=np.float64)
    sums = np.zeros(ACTION_DIM, dtype=np.float64)
    square_sums = np.zeros(ACTION_DIM, dtype=np.float64)
    median_rows: list[np.ndarray] = []
    count = 0
    global_row = 0

    for states in states_list:
        anchors = np.arange(len(states))
        for offset in range(1, chunk_size + 1):
            indices = np.minimum(anchors + offset, len(states) - 1)
            future = states[indices]
            values = future - states
            values[:, GRIPPER_DIMS] = future[:, GRIPPER_DIMS]
            values64 = values.astype(np.float64, copy=False)
            minimum = np.minimum(minimum, values64.min(axis=0))
            maximum = np.maximum(maximum, values64.max(axis=0))
            sums += values64.sum(axis=0)
            square_sums += np.square(values64).sum(axis=0)
            count += len(values64)

            first = (-global_row) % median_stride
            if first < len(values):
                median_rows.append(values[first::median_stride].copy())
            global_row += len(values)

    mean = sums / count
    variance = np.maximum(square_sums / count - np.square(mean), 0.0)
    median_sample = np.concatenate(median_rows, axis=0)
    return {
        "actions_min": minimum.astype(np.float32),
        "actions_max": maximum.astype(np.float32),
        "actions_mean": mean.astype(np.float32),
        "actions_std": np.sqrt(variance).astype(np.float32),
        "actions_median": np.median(median_sample, axis=0).astype(np.float32),
    }


def load_or_compute_q0_action_statistics(
    data_dir: str | Path,
    trajectories: list[np.ndarray],
    chunk_size: int = CHUNK_SIZE,
) -> dict[str, np.ndarray]:
    """Load contract-specific statistics or compute and persist them."""
    path = Path(data_dir) / "q0_action_statistics.json"
    if path.exists():
        payload = json.loads(path.read_text())
        if payload.get("contract") != CONTRACT_NAME or payload.get("chunk_size") != chunk_size:
            raise ValueError(f"stale or incompatible q0 statistics: {path}")
        return {key: np.asarray(value, dtype=np.float32) for key, value in payload["statistics"].items()}

    statistics = calculate_q0_action_statistics(trajectories, chunk_size=chunk_size)
    payload = {
        "contract": CONTRACT_NAME,
        "chunk_size": chunk_size,
        "first_future_offset": 1,
        "gripper_mode": "absolute_measured_joint_position_rad",
        "statistics": {key: value.tolist() for key, value in statistics.items()},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return statistics

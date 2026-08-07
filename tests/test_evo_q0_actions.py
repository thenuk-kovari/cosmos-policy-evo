import numpy as np

from cosmos_policy.datasets.evo_q0_actions import (
    GRIPPER_DIMS,
    build_q0_anchored_chunk,
    calculate_q0_action_statistics,
    future_indices,
    normalize_action_chunk,
)


def trajectory(length: int = 6) -> np.ndarray:
    states = np.zeros((length, 17), dtype=np.float32)
    t = np.arange(length, dtype=np.float32)
    states[:, 0] = t
    states[:, 1] = 10.0 * t
    states[:, 7] = -0.9 + 0.1 * t
    states[:, 8] = -2.0 * t
    states[:, 15] = -0.8 + 0.05 * t
    states[:, 16] = 0.4 + 0.01 * t
    return states


def test_chunk_uses_one_fixed_anchor_and_starts_at_next_frame():
    states = trajectory()
    chunk = build_q0_anchored_chunk(states, anchor_index=1, chunk_size=3)

    # Fixed q0: offsets are q2-q1, q3-q1, q4-q1. They are not the
    # per-step sequence q2-q1, q3-q2, q4-q3.
    np.testing.assert_allclose(chunk[:, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(chunk[:, 1], [10.0, 20.0, 30.0])
    np.testing.assert_allclose(chunk[:, 8], [-2.0, -4.0, -6.0])
    np.testing.assert_allclose(chunk[:, 16], [0.01, 0.02, 0.03], atol=1e-6)


def test_grippers_are_future_absolute_measured_radians():
    states = trajectory()
    chunk = build_q0_anchored_chunk(states, anchor_index=1, chunk_size=3)
    np.testing.assert_allclose(chunk[:, 7], states[2:5, 7])
    np.testing.assert_allclose(chunk[:, 15], states[2:5, 15])


def test_terminal_padding_repeats_final_absolute_target():
    states = trajectory()
    np.testing.assert_array_equal(future_indices(4, 4, len(states)), [5, 5, 5, 5])
    chunk = build_q0_anchored_chunk(states, anchor_index=4, chunk_size=4)
    np.testing.assert_allclose(chunk, np.repeat(chunk[:1], 4, axis=0))


def test_statistics_are_over_generated_chunks_not_absolute_joint_states():
    states = trajectory(length=4)
    stats = calculate_q0_action_statistics([states], chunk_size=2, max_median_samples=100)
    # Largest fixed-anchor joint-0 displacement over a two-row chunk is 2,
    # whereas the stored absolute trajectory reaches 3.
    assert stats["actions_max"][0] == 2.0
    assert stats["actions_min"][0] == 0.0
    # Grippers are absolute, so their statistics remain in raw joint radians.
    assert stats["actions_min"][GRIPPER_DIMS[0]] < 0.0


def test_normalization_round_trip_formula():
    states = trajectory(length=5)
    chunk = build_q0_anchored_chunk(states, anchor_index=0, chunk_size=3)
    stats = calculate_q0_action_statistics([states], chunk_size=3, max_median_samples=100)
    normalized = normalize_action_chunk(chunk, stats)
    span = stats["actions_max"] - stats["actions_min"]
    denominator = np.where(np.abs(span) < 1e-8, 1.0, span)
    restored = 0.5 * (normalized + 1.0) * denominator + stats["actions_min"]
    np.testing.assert_allclose(restored, chunk, atol=1e-6)

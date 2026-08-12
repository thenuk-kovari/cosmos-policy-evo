import numpy as np

from cosmos_policy.datasets.ee_q0_actions import (
    ELEVATOR_DELTA,
    LEFT_EE_TRANSLATION,
    LEFT_GRIPPER,
    LEFT_JOINT_DELTA,
    NORMALIZATION_CONTRACT,
    RIGHT_EE_TRANSLATION,
    RIGHT_GRIPPER,
    RIGHT_JOINT_DELTA,
    SHARED_ACTION_DIM,
    build_umi_shared_action_chunk,
    build_shared_pose_source,
    build_umi_shared_action_chunk_from_storage,
    canonical_shared_statistics,
    normalize_shared_action_chunk,
    shared_q0_action_extrema,
    decode_q0_relative_pose,
    encode_q0_relative_pose,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)


def _rotz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _matrix_to_quaternion_xyzw_z(theta):
    return np.array([0.0, 0.0, np.sin(theta / 2), np.cos(theta / 2)])


def test_rotation_6d_identity_layout_and_roundtrip():
    rotation = np.stack([_rotz(0.0), _rotz(0.7), _rotz(-1.2)])
    six = matrix_to_rotation_6d(rotation)
    np.testing.assert_allclose(six[0], [1, 0, 0, 0, 1, 0])
    np.testing.assert_allclose(rotation_6d_to_matrix(six), rotation, atol=1e-12)


def test_q0_pose_roundtrip_and_world_transform_invariance():
    p0 = np.array([1.0, -2.0, 0.3])
    r0 = _rotz(0.4)
    positions = np.array([[1.1, -1.9, 0.4], [0.8, -1.7, 0.5]])
    rotations = np.stack([_rotz(0.6), _rotz(-0.1)])
    dp, dr = encode_q0_relative_pose(p0, r0, positions, rotations)
    decoded_p, decoded_r = decode_q0_relative_pose(p0, r0, dp, dr)
    np.testing.assert_allclose(decoded_p, positions, atol=1e-12)
    np.testing.assert_allclose(decoded_r, rotations, atol=1e-12)

    world_rotation = _rotz(-1.1)
    world_translation = np.array([4.0, 2.0, -0.5])
    transformed_p0 = world_rotation @ p0 + world_translation
    transformed_positions = np.einsum("ij,tj->ti", world_rotation, positions) + world_translation
    transformed_r0 = world_rotation @ r0
    transformed_rotations = np.einsum("ij,tjk->tik", world_rotation, rotations)
    transformed_dp, transformed_dr = encode_q0_relative_pose(
        transformed_p0, transformed_r0, transformed_positions, transformed_rotations
    )
    np.testing.assert_allclose(transformed_dp, dp, atol=1e-12)
    np.testing.assert_allclose(transformed_dr, dr, atol=1e-12)


def test_umi_chunk_uses_shared_35d_mask_and_absolute_grippers():
    length = 60
    t = np.arange(length, dtype=np.float64)
    left_p = np.stack((0.01 * t, np.zeros(length), np.zeros(length)), axis=1)
    right_p = np.stack((np.zeros(length), -0.01 * t, np.zeros(length)), axis=1)
    left_q = np.stack([_matrix_to_quaternion_xyzw_z(0.01 * x) for x in t])
    right_q = np.stack([_matrix_to_quaternion_xyzw_z(-0.02 * x) for x in t])
    left_g = np.linspace(0.0, 0.103, length)
    right_g = np.linspace(0.103, 0.0, length)
    action, mask = build_umi_shared_action_chunk(
        left_p, left_q, left_g, right_p, right_q, right_g, anchor_index=3
    )
    assert action.shape == (50, SHARED_ACTION_DIM)
    assert mask.shape == (SHARED_ACTION_DIM,)
    assert mask.sum() == 20
    assert np.all(mask[LEFT_JOINT_DELTA] == 0)
    assert np.all(mask[RIGHT_JOINT_DELTA] == 0)
    assert mask[ELEVATOR_DELTA] == 0
    assert mask[LEFT_GRIPPER] == mask[RIGHT_GRIPPER] == 1
    np.testing.assert_allclose(action[0, LEFT_GRIPPER], left_g[4] / 0.103)
    np.testing.assert_allclose(action[0, RIGHT_GRIPPER], right_g[4] / 0.103)
    np.testing.assert_allclose(action[:, LEFT_JOINT_DELTA], 0)
    np.testing.assert_allclose(action[:, RIGHT_JOINT_DELTA], 0)


def test_absolute_storage_reconstructs_same_q0_chunk():
    length = 55
    t = np.arange(length, dtype=np.float64)
    left_p = np.stack((0.002 * t, -0.001 * t, np.zeros(length)), axis=1)
    right_p = np.stack((-0.001 * t, np.zeros(length), 0.001 * t), axis=1)
    left_q = np.stack([_matrix_to_quaternion_xyzw_z(0.005 * x) for x in t])
    right_q = np.stack([_matrix_to_quaternion_xyzw_z(-0.007 * x) for x in t])
    left_g = np.linspace(0.01, 0.09, length)
    right_g = np.linspace(0.09, 0.01, length)
    direct, direct_mask = build_umi_shared_action_chunk(
        left_p, left_q, left_g, right_p, right_q, right_g, anchor_index=2
    )
    source = build_shared_pose_source(left_p, left_q, left_g, right_p, right_q, right_g)
    stored, stored_mask = build_umi_shared_action_chunk_from_storage(source, anchor_index=2)
    np.testing.assert_allclose(stored, direct, atol=1e-6)
    np.testing.assert_array_equal(stored_mask, direct_mask)


def test_fixed_statistics_keep_masked_zero_channels_at_zero():
    statistics = canonical_shared_statistics()
    assert NORMALIZATION_CONTRACT == "bimanual_shared35_fixed_physical_translation1m_v2"
    for translation in (LEFT_EE_TRANSLATION, RIGHT_EE_TRANSLATION):
        np.testing.assert_allclose(statistics["actions_min"][translation], -1.0)
        np.testing.assert_allclose(statistics["actions_max"][translation], 1.0)

    chunk = np.zeros((50, SHARED_ACTION_DIM), dtype=np.float32)
    normalized = normalize_shared_action_chunk(chunk, statistics)
    np.testing.assert_allclose(normalized[:, LEFT_JOINT_DELTA], 0)
    np.testing.assert_allclose(normalized[:, RIGHT_JOINT_DELTA], 0)
    np.testing.assert_allclose(normalized[:, ELEVATOR_DELTA], 0)
    # Absolute closed gripper maps to -1 under the canonical [0,1] range.
    np.testing.assert_allclose(normalized[:, [LEFT_GRIPPER, RIGHT_GRIPPER]], -1)


def test_vectorized_extrema_match_brute_force_chunks():
    length = 12
    t = np.arange(length, dtype=np.float64)
    p0 = np.stack((0.01 * t, 0.002 * t, np.zeros(length)), axis=1)
    p1 = np.stack((-0.003 * t, 0.004 * t, 0.001 * t), axis=1)
    q0 = np.stack([_matrix_to_quaternion_xyzw_z(0.02 * x) for x in t])
    q1 = np.stack([_matrix_to_quaternion_xyzw_z(-0.01 * x) for x in t])
    source = build_shared_pose_source(
        p0, q0, np.linspace(0, 0.103, length),
        p1, q1, np.linspace(0.103, 0, length),
    )
    exact_min, exact_max = shared_q0_action_extrema(source, chunk_size=5)
    chunks = [build_umi_shared_action_chunk_from_storage(source, i, chunk_size=5)[0] for i in range(length)]
    rows = np.concatenate(chunks, axis=0)
    np.testing.assert_allclose(exact_min, rows.min(axis=0), atol=1e-6)
    np.testing.assert_allclose(exact_max, rows.max(axis=0), atol=1e-6)

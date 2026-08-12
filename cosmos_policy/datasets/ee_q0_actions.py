"""Shared bimanual q0-anchored EE action contract.

The contract is deliberately explicit about rotation layout and reference
frames.  Translations and rotations are expressed in each hand's query-time
tool frame, which removes an arbitrary constant VIO world transform::

    dp[k] = R0.T @ (p[k] - p0)
    dR[k] = R0.T @ R[k]

The 6-D rotation stores the first two *columns* of ``dR`` as
``[c0x,c0y,c0z,c1x,c1y,c1z]``.  It is not an additive angular delta.
"""

from __future__ import annotations

import numpy as np


CHUNK_SIZE = 50
EE_ACTION_DIM = 18
SHARED_ACTION_DIM = 35

LEFT_EE_TRANSLATION = slice(0, 3)
LEFT_EE_ROTATION_6D = slice(3, 9)
RIGHT_EE_TRANSLATION = slice(9, 12)
RIGHT_EE_ROTATION_6D = slice(12, 18)
LEFT_JOINT_DELTA = slice(18, 25)
LEFT_GRIPPER = 25
RIGHT_JOINT_DELTA = slice(26, 33)
RIGHT_GRIPPER = 33
ELEVATOR_DELTA = 34

UMI_ACTIVE_DIMS = np.array(
    [*range(EE_ACTION_DIM), LEFT_GRIPPER, RIGHT_GRIPPER], dtype=np.int64
)

CONTRACT_NAME = "bimanual_q0_body_ee6d_joint35_v1"
STORAGE_CONTRACT = "bimanual_absolute_world_pose_ee6d_source_v1"
NORMALIZATION_CONTRACT = "bimanual_shared35_fixed_physical_translation1m_v2"
ROTATION_6D_LAYOUT = "first_two_columns_c0_then_c1"


def normalize_quaternion_xyzw(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape[-1] != 4:
        raise ValueError(f"expected quaternion [...,4], got {quaternion.shape}")
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm < 1e-8) or not np.isfinite(norm).all():
        raise ValueError("invalid zero or non-finite quaternion")
    return quaternion / norm


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert normalized-or-unnormalized xyzw quaternions to matrices."""
    q = normalize_quaternion_xyzw(quaternion)
    x, y, z, w = np.moveaxis(q, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.stack(
        (
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
        ),
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"expected rotation matrix [...,3,3], got {matrix.shape}")
    return np.concatenate((matrix[..., :, 0], matrix[..., :, 1]), axis=-1)


def rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    """Project two predicted columns onto SO(3) with Gram--Schmidt."""
    rotation_6d = np.asarray(rotation_6d, dtype=np.float64)
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f"expected rotation 6D [...,6], got {rotation_6d.shape}")
    a1, a2 = rotation_6d[..., :3], rotation_6d[..., 3:]
    n1 = np.linalg.norm(a1, axis=-1, keepdims=True)
    if np.any(n1 < 1e-8):
        raise ValueError("degenerate first rotation-6D column")
    b1 = a1 / n1
    a2_orthogonal = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    n2 = np.linalg.norm(a2_orthogonal, axis=-1, keepdims=True)
    if np.any(n2 < 1e-8):
        raise ValueError("degenerate second rotation-6D column")
    b2 = a2_orthogonal / n2
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-1)


def encode_q0_relative_pose(
    query_position: np.ndarray,
    query_rotation: np.ndarray,
    future_positions: np.ndarray,
    future_rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode future poses in the frozen query-time body frame."""
    p0 = np.asarray(query_position, dtype=np.float64)
    r0 = np.asarray(query_rotation, dtype=np.float64)
    pt = np.asarray(future_positions, dtype=np.float64)
    rt = np.asarray(future_rotations, dtype=np.float64)
    if p0.shape != (3,) or r0.shape != (3, 3):
        raise ValueError("query pose must be position [3] and rotation [3,3]")
    if pt.ndim != 2 or pt.shape[1] != 3 or rt.shape != (len(pt), 3, 3):
        raise ValueError("future poses must be positions [T,3] and rotations [T,3,3]")
    translation = np.einsum("ij,tj->ti", r0.T, pt - p0)
    relative_rotation = np.einsum("ij,tjk->tik", r0.T, rt)
    return translation, matrix_to_rotation_6d(relative_rotation)


def decode_q0_relative_pose(
    query_position: np.ndarray,
    query_rotation: np.ndarray,
    translation: np.ndarray,
    rotation_6d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode a q0-relative pose chunk back into the source world frame."""
    p0 = np.asarray(query_position, dtype=np.float64)
    r0 = np.asarray(query_rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    relative_rotation = rotation_6d_to_matrix(rotation_6d)
    position = p0 + np.einsum("ij,tj->ti", r0, translation)
    rotation = np.einsum("ij,tjk->tik", r0, relative_rotation)
    return position, rotation


def canonical_gripper_aperture(width_m: np.ndarray, maximum_width_m: float = 0.103) -> np.ndarray:
    """Map DAS physical opening width to the shared closed=0/open=1 scalar."""
    if maximum_width_m <= 0:
        raise ValueError("maximum gripper width must be positive")
    width = np.asarray(width_m, dtype=np.float64)
    if not np.isfinite(width).all():
        raise ValueError("gripper width contains NaN or infinity")
    return np.clip(width / maximum_width_m, 0.0, 1.0)


def future_indices(anchor_index: int, length: int, chunk_size: int = CHUNK_SIZE) -> np.ndarray:
    if not 0 <= anchor_index < length:
        raise IndexError(f"anchor {anchor_index} outside trajectory length {length}")
    return np.minimum(anchor_index + 1 + np.arange(chunk_size), length - 1)


def build_umi_shared_action_chunk(
    left_positions: np.ndarray,
    left_quaternions_xyzw: np.ndarray,
    left_gripper_width_m: np.ndarray,
    right_positions: np.ndarray,
    right_quaternions_xyzw: np.ndarray,
    right_gripper_width_m: np.ndarray,
    anchor_index: int,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a masked 35-D UMI chunk; joint/elevator entries remain zero."""
    arrays = [
        np.asarray(left_positions), np.asarray(left_quaternions_xyzw),
        np.asarray(left_gripper_width_m), np.asarray(right_positions),
        np.asarray(right_quaternions_xyzw), np.asarray(right_gripper_width_m),
    ]
    length = len(arrays[0])
    if any(len(x) != length for x in arrays):
        raise ValueError("all synchronized UMI streams must have equal length")
    indices = future_indices(anchor_index, length, chunk_size)
    left_rotation = quaternion_xyzw_to_matrix(arrays[1])
    right_rotation = quaternion_xyzw_to_matrix(arrays[4])
    left_dp, left_dr = encode_q0_relative_pose(
        arrays[0][anchor_index], left_rotation[anchor_index], arrays[0][indices], left_rotation[indices]
    )
    right_dp, right_dr = encode_q0_relative_pose(
        arrays[3][anchor_index], right_rotation[anchor_index], arrays[3][indices], right_rotation[indices]
    )
    action = np.zeros((chunk_size, SHARED_ACTION_DIM), dtype=np.float32)
    action[:, LEFT_EE_TRANSLATION] = left_dp
    action[:, LEFT_EE_ROTATION_6D] = left_dr
    action[:, RIGHT_EE_TRANSLATION] = right_dp
    action[:, RIGHT_EE_ROTATION_6D] = right_dr
    action[:, LEFT_GRIPPER] = canonical_gripper_aperture(arrays[2][indices])
    action[:, RIGHT_GRIPPER] = canonical_gripper_aperture(arrays[5][indices])
    loss_mask = np.zeros(SHARED_ACTION_DIM, dtype=np.float32)
    loss_mask[UMI_ACTIVE_DIMS] = 1.0
    return action, loss_mask


def validate_shared_pose_source(source: np.ndarray) -> np.ndarray:
    """Validate per-frame absolute-pose storage used by the dataset loader."""
    source = np.asarray(source)
    if source.ndim != 2 or source.shape[1] != SHARED_ACTION_DIM:
        raise ValueError(f"expected shared pose source [T,{SHARED_ACTION_DIM}], got {source.shape}")
    if len(source) == 0 or not np.isfinite(source).all():
        raise ValueError("shared pose source is empty or non-finite")
    # Fail early if either stored 6-D orientation is degenerate.
    rotation_6d_to_matrix(source[:, LEFT_EE_ROTATION_6D])
    rotation_6d_to_matrix(source[:, RIGHT_EE_ROTATION_6D])
    return source.astype(np.float32, copy=False)


def build_shared_pose_source(
    left_positions: np.ndarray,
    left_quaternions_xyzw: np.ndarray,
    left_gripper_width_m: np.ndarray,
    right_positions: np.ndarray,
    right_quaternions_xyzw: np.ndarray,
    right_gripper_width_m: np.ndarray,
) -> np.ndarray:
    """Pack synchronized absolute world poses into the 35-D storage layout.

    This is storage, not the learned action. The learned chunk is constructed
    at sample time by :func:`build_umi_shared_action_chunk_from_storage`.
    """
    arrays = [
        np.asarray(left_positions), np.asarray(left_quaternions_xyzw),
        np.asarray(left_gripper_width_m), np.asarray(right_positions),
        np.asarray(right_quaternions_xyzw), np.asarray(right_gripper_width_m),
    ]
    length = len(arrays[0])
    if length == 0 or any(len(value) != length for value in arrays):
        raise ValueError("all synchronized source streams must have the same non-zero length")
    if arrays[0].shape != (length, 3) or arrays[3].shape != (length, 3):
        raise ValueError("positions must have shape [T,3]")
    if arrays[1].shape != (length, 4) or arrays[4].shape != (length, 4):
        raise ValueError("quaternions must have shape [T,4] in xyzw order")
    source = np.zeros((length, SHARED_ACTION_DIM), dtype=np.float32)
    source[:, LEFT_EE_TRANSLATION] = arrays[0]
    source[:, LEFT_EE_ROTATION_6D] = matrix_to_rotation_6d(quaternion_xyzw_to_matrix(arrays[1]))
    source[:, RIGHT_EE_TRANSLATION] = arrays[3]
    source[:, RIGHT_EE_ROTATION_6D] = matrix_to_rotation_6d(quaternion_xyzw_to_matrix(arrays[4]))
    source[:, LEFT_GRIPPER] = canonical_gripper_aperture(arrays[2])
    source[:, RIGHT_GRIPPER] = canonical_gripper_aperture(arrays[5])
    return source


def build_umi_shared_action_chunk_from_storage(
    source: np.ndarray,
    anchor_index: int,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the q0-local UMI action chunk from absolute pose storage."""
    source = validate_shared_pose_source(source)
    indices = future_indices(anchor_index, len(source), chunk_size)
    left_rotation = rotation_6d_to_matrix(source[:, LEFT_EE_ROTATION_6D])
    right_rotation = rotation_6d_to_matrix(source[:, RIGHT_EE_ROTATION_6D])
    left_dp, left_dr = encode_q0_relative_pose(
        source[anchor_index, LEFT_EE_TRANSLATION],
        left_rotation[anchor_index],
        source[indices, LEFT_EE_TRANSLATION],
        left_rotation[indices],
    )
    right_dp, right_dr = encode_q0_relative_pose(
        source[anchor_index, RIGHT_EE_TRANSLATION],
        right_rotation[anchor_index],
        source[indices, RIGHT_EE_TRANSLATION],
        right_rotation[indices],
    )
    action = np.zeros((chunk_size, SHARED_ACTION_DIM), dtype=np.float32)
    action[:, LEFT_EE_TRANSLATION] = left_dp
    action[:, LEFT_EE_ROTATION_6D] = left_dr
    action[:, RIGHT_EE_TRANSLATION] = right_dp
    action[:, RIGHT_EE_ROTATION_6D] = right_dr
    action[:, LEFT_GRIPPER] = source[indices, LEFT_GRIPPER]
    action[:, RIGHT_GRIPPER] = source[indices, RIGHT_GRIPPER]
    mask = np.zeros(SHARED_ACTION_DIM, dtype=np.float32)
    mask[UMI_ACTIVE_DIMS] = 1.0
    return action, mask


def canonical_shared_statistics() -> dict[str, np.ndarray]:
    """Return immutable physical normalization bounds for both training stages.

    Fixed bounds are intentional: stage two must not reinterpret the stage-one
    head merely because a different embodiment has different empirical extrema.
    """
    action_min = np.zeros(SHARED_ACTION_DIM, dtype=np.float32)
    action_max = np.zeros(SHARED_ACTION_DIM, dtype=np.float32)
    for translation in (LEFT_EE_TRANSLATION, RIGHT_EE_TRANSLATION):
        action_min[translation], action_max[translation] = -1.0, 1.0
    for rotation in (LEFT_EE_ROTATION_6D, RIGHT_EE_ROTATION_6D):
        action_min[rotation], action_max[rotation] = -1.0, 1.0
    for joints in (LEFT_JOINT_DELTA, RIGHT_JOINT_DELTA):
        action_min[joints], action_max[joints] = -2.0, 2.0
    action_min[[LEFT_GRIPPER, RIGHT_GRIPPER]] = 0.0
    action_max[[LEFT_GRIPPER, RIGHT_GRIPPER]] = 1.0
    action_min[ELEVATOR_DELTA], action_max[ELEVATOR_DELTA] = -0.7, 0.7

    proprio_min = np.full(17, -np.pi, dtype=np.float32)
    proprio_max = np.full(17, np.pi, dtype=np.float32)
    proprio_min[[7, 15]], proprio_max[[7, 15]] = 0.0, 1.0
    proprio_min[16], proprio_max[16] = -0.7, 0.7

    def statistics(minimum: np.ndarray, maximum: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
        midpoint = (minimum + maximum) / 2.0
        return {
            f"{prefix}_min": minimum,
            f"{prefix}_max": maximum,
            f"{prefix}_mean": midpoint.copy(),
            f"{prefix}_std": ((maximum - minimum) / np.sqrt(12.0)).astype(np.float32),
            f"{prefix}_median": midpoint.copy(),
        }

    return {**statistics(action_min, action_max, "actions"), **statistics(proprio_min, proprio_max, "proprio")}


def normalize_shared_action_chunk(chunk: np.ndarray, statistics: dict[str, np.ndarray]) -> np.ndarray:
    chunk = np.asarray(chunk, dtype=np.float32)
    minimum = np.asarray(statistics["actions_min"], dtype=np.float32)
    maximum = np.asarray(statistics["actions_max"], dtype=np.float32)
    if chunk.shape[-1] != SHARED_ACTION_DIM or minimum.shape != (SHARED_ACTION_DIM,) or maximum.shape != (SHARED_ACTION_DIM,):
        raise ValueError("shared action chunk/statistics must use the 35-D contract")
    denominator = maximum - minimum
    if np.any(denominator <= 0):
        raise ValueError("normalization bounds must have positive span")
    return (2.0 * ((chunk - minimum) / denominator) - 1.0).astype(np.float32)


def shared_q0_action_extrema(
    source: np.ndarray, chunk_size: int = CHUNK_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact per-channel extrema over every anchor/row efficiently."""
    source = validate_shared_pose_source(source)
    anchors = np.arange(len(source))
    left_rotation = rotation_6d_to_matrix(source[:, LEFT_EE_ROTATION_6D])
    right_rotation = rotation_6d_to_matrix(source[:, RIGHT_EE_ROTATION_6D])
    minimum = np.full(SHARED_ACTION_DIM, np.inf, dtype=np.float64)
    maximum = np.full(SHARED_ACTION_DIM, -np.inf, dtype=np.float64)
    for offset in range(1, chunk_size + 1):
        indices = np.minimum(anchors + offset, len(source) - 1)
        values = np.zeros((len(source), SHARED_ACTION_DIM), dtype=np.float64)
        for position_slice, rotation_slice, rotation in (
            (LEFT_EE_TRANSLATION, LEFT_EE_ROTATION_6D, left_rotation),
            (RIGHT_EE_TRANSLATION, RIGHT_EE_ROTATION_6D, right_rotation),
        ):
            values[:, position_slice] = np.einsum(
                "tji,tj->ti",
                rotation,
                source[indices, position_slice] - source[:, position_slice],
            )
            relative = np.einsum("tji,tjk->tik", rotation, rotation[indices])
            values[:, rotation_slice] = matrix_to_rotation_6d(relative)
        values[:, LEFT_GRIPPER] = source[indices, LEFT_GRIPPER]
        values[:, RIGHT_GRIPPER] = source[indices, RIGHT_GRIPPER]
        minimum = np.minimum(minimum, values.min(axis=0))
        maximum = np.maximum(maximum, values.max(axis=0))
    return minimum.astype(np.float32), maximum.astype(np.float32)

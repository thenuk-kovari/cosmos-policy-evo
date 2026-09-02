"""Evo teleop loader for the shared 35-D EE + joint q0 action head."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from cosmos_policy.datasets.aloha_dataset import ALOHADataset
from cosmos_policy.datasets.dataset_utils import rescale_data
from cosmos_policy.datasets.ee_q0_actions import (
    CHUNK_SIZE,
    CONTRACT_NAME,
    EVO_STORAGE_CONTRACT,
    EVO_TRANSFER_NORMALIZATION_CONTRACT,
    NORMALIZATION_CONTRACT,
    SHARED_ACTION_DIM,
    build_evo_shared_action_chunk_from_storage,
    evo_transfer_shared_statistics,
    normalize_shared_action_chunk,
)


PROPRIO_DIM = 17


class EvoEEJointQ0Dataset(ALOHADataset):
    """Supervise every EE, joint, gripper, and elevator dimension on Evo.

    HDF5 ``action`` stores absolute per-frame base-frame palm poses and measured
    joint state. It is not the learned action. The learned 50-row chunk is
    constructed from future measured states relative to one frozen query-time
    anchor by :func:`build_evo_shared_action_chunk_from_storage`.
    """

    def __init__(self, *args, **kwargs):
        requested_normalize_actions = bool(kwargs.pop("normalize_actions", True))
        requested_normalize_proprio = bool(kwargs.pop("normalize_proprio", True))
        configured_data_dir = Path(kwargs.get("data_dir", args[0] if args else "."))
        statistics_dir = Path(kwargs.pop("statistics_dir", configured_data_dir))
        chunk_size = int(kwargs.get("chunk_size", CHUNK_SIZE))
        if chunk_size != CHUNK_SIZE:
            raise ValueError(f"{CONTRACT_NAME} requires chunk_size={CHUNK_SIZE}, got {chunk_size}")

        # Absolute pose/joint storage must never be normalized by the base
        # class. Normalize only the constructed q0 chunks below.
        super().__init__(*args, normalize_actions=False, normalize_proprio=False, **kwargs)
        self.normalize_shared_actions = requested_normalize_actions
        self.normalize_shared_proprio = requested_normalize_proprio

        stats_path = statistics_dir / "dataset_statistics.json"
        if not stats_path.is_file():
            raise FileNotFoundError(f"missing canonical statistics: {stats_path}")
        self.dataset_stats = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in json.loads(stats_path.read_text()).items()
        }
        reference_path = statistics_dir / "evo_q0_reference_statistics.json"
        if not reference_path.is_file():
            raise FileNotFoundError(f"missing Evo-only q0 reference statistics: {reference_path}")
        reference_statistics = json.loads(reference_path.read_text())
        expected_statistics = evo_transfer_shared_statistics(reference_statistics)
        for key, expected in expected_statistics.items():
            actual = np.asarray(self.dataset_stats.get(key), dtype=np.float32)
            if actual.shape != expected.shape or not np.array_equal(actual, expected):
                raise ValueError(
                    f"{stats_path}: {key} does not match {EVO_TRANSFER_NORMALIZATION_CONTRACT}"
                )

        expected_action_mask = np.ones(SHARED_ACTION_DIM, dtype=np.float32)
        expected_proprio_mask = np.ones(PROPRIO_DIM, dtype=np.float32)
        for episode in self.data.values():
            if episode["actions"].shape[1] != SHARED_ACTION_DIM:
                raise ValueError(f"expected 35-D absolute source, got {episode['actions'].shape}")
            if episode["proprio"].shape[1] != PROPRIO_DIM:
                raise ValueError(f"expected 17-D proprio, got {episode['proprio'].shape}")
            if not np.array_equal(episode["action_dim_mask"], expected_action_mask):
                raise ValueError(f"{episode['file_path']}: Evo must supervise all 35 action dimensions")
            if not np.array_equal(episode["proprio_dim_mask"], expected_proprio_mask):
                raise ValueError(f"{episode['file_path']}: Evo must expose all 17 proprio dimensions")
            self._validate_hdf5_contract(episode["file_path"])

        if self.normalize_shared_proprio:
            self.data = rescale_data(self.data, self.dataset_stats, "proprio")
            self.rollout_data = rescale_data(self.rollout_data, self.dataset_stats, "proprio")

    @staticmethod
    def _validate_hdf5_contract(path: str) -> None:
        with h5py.File(path, "r") as handle:
            def attr(name: str) -> str:
                value = handle.attrs.get(name, "")
                return value.decode() if isinstance(value, bytes) else str(value)

            expected = {
                "action_contract": CONTRACT_NAME,
                "action_storage_contract": EVO_STORAGE_CONTRACT,
                "source_pose_link": "base_link",
                "fk_root": "base_link",
                "fk_tips": "left_palm,right_palm",
                "position_post_transform": "none",
                "orientation_post_transform": "right_multiply_q_offset_wxyz",
                "delta_reference_frame": "query_body",
                "recorded_action_column_used": "False",
            }
            for name, wanted in expected.items():
                if attr(name) != wanted:
                    raise ValueError(f"{path}: {name}={attr(name)!r}, expected {wanted!r}")

    def __getitem__(self, idx: int) -> dict:
        sample = super().__getitem__(idx)
        # A normalized error e corresponds to e * (max-min)/2 physical units.
        # Carry this immutable scale into the model only for diagnostics; it is
        # not an input and does not alter the optimization objective.
        sample["action_denormalization_scale"] = (
            (self.dataset_stats["actions_max"] - self.dataset_stats["actions_min"]) / 2.0
        ).astype(np.float32)
        return sample

    def _get_action_chunk(self, episode_data: dict, relative_step_idx: int) -> np.ndarray:
        chunk, mask = build_evo_shared_action_chunk_from_storage(
            episode_data["actions"], relative_step_idx, self.chunk_size
        )
        if not np.array_equal(mask, episode_data["action_dim_mask"]):
            raise RuntimeError("constructed Evo mask differs from stored contract")
        if self.normalize_shared_actions:
            chunk = normalize_shared_action_chunk(chunk, self.dataset_stats)
        return chunk

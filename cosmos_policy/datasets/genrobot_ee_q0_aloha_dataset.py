"""ALOHA-compatible loader for GenRobot q0-local EE-6D actions."""

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
    SHARED_ACTION_DIM,
    STORAGE_CONTRACT,
    NORMALIZATION_CONTRACT,
    UMI_ACTIVE_DIMS,
    build_umi_shared_action_chunk_from_storage,
    canonical_shared_statistics,
    normalize_shared_action_chunk,
)


PROPRIO_DIM = 17


class GenRobotEEQ0Dataset(ALOHADataset):
    """Build 50-step fixed-q0 EE chunks from synchronized absolute VIO poses."""

    def __init__(self, *args, **kwargs):
        requested_normalize_actions = bool(kwargs.pop("normalize_actions", True))
        requested_normalize_proprio = bool(kwargs.pop("normalize_proprio", True))
        configured_data_dir = Path(kwargs.get("data_dir", args[0] if args else "."))
        statistics_dir = Path(kwargs.pop("statistics_dir", configured_data_dir))
        chunk_size = int(kwargs.get("chunk_size", CHUNK_SIZE))
        if chunk_size != CHUNK_SIZE:
            raise ValueError(f"{CONTRACT_NAME} requires chunk_size={CHUNK_SIZE}, got {chunk_size}")

        # Stored actions are absolute VIO pose sources, so the base class must
        # never normalize them. We construct and normalize q0 chunks below.
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
        expected = canonical_shared_statistics()
        for key, expected_value in expected.items():
            actual = np.asarray(self.dataset_stats.get(key), dtype=np.float32)
            if actual.shape != expected_value.shape or not np.allclose(actual, expected_value, atol=1e-7):
                raise ValueError(f"{stats_path}: {key} does not match {NORMALIZATION_CONTRACT}")

        expected_mask = np.zeros(SHARED_ACTION_DIM, dtype=np.float32)
        expected_mask[UMI_ACTIVE_DIMS] = 1.0
        expected_proprio_mask = np.zeros(PROPRIO_DIM, dtype=np.float32)
        expected_proprio_mask[[7, 15]] = 1.0
        for episode in self.data.values():
            if episode["actions"].shape[1] != SHARED_ACTION_DIM:
                raise ValueError(f"expected 35-D absolute pose storage, got {episode['actions'].shape}")
            if episode["proprio"].shape[1] != PROPRIO_DIM:
                raise ValueError(f"expected 17-D shared proprio, got {episode['proprio'].shape}")
            if not np.array_equal(episode["action_dim_mask"], expected_mask):
                raise ValueError(f"{episode['file_path']}: wrong UMI action availability mask")
            if not np.array_equal(episode["proprio_dim_mask"], expected_proprio_mask):
                raise ValueError(f"{episode['file_path']}: wrong UMI proprio availability mask")
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

            if attr("action_contract") != CONTRACT_NAME:
                raise ValueError(f"{path}: wrong action_contract")
            if attr("action_storage_contract") != STORAGE_CONTRACT:
                raise ValueError(f"{path}: wrong action_storage_contract")
            if attr("source_pose_link") != "base_link":
                raise ValueError(f"{path}: source_pose_link must be explicit base_link")
            if attr("delta_reference_frame") != "query_body":
                raise ValueError(f"{path}: delta_reference_frame must be query_body")

    def _get_action_chunk(self, episode_data: dict, relative_step_idx: int) -> np.ndarray:
        chunk, mask = build_umi_shared_action_chunk_from_storage(
            episode_data["actions"], relative_step_idx, self.chunk_size
        )
        if not np.array_equal(mask, episode_data["action_dim_mask"]):
            raise RuntimeError("constructed UMI mask differs from stored contract")
        if self.normalize_shared_actions:
            chunk = normalize_shared_action_chunk(chunk, self.dataset_stats)
        return chunk

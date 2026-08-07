"""ALOHA-compatible loader for EVO query-anchored observed-state actions."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from cosmos_policy.datasets.aloha_dataset import ALOHADataset
from cosmos_policy.datasets.dataset_utils import rescale_data
from cosmos_policy.datasets.evo_q0_actions import (
    ACTION_DIM,
    CHUNK_SIZE,
    CONTRACT_NAME,
    PROPRIO_DIM,
    build_q0_anchored_chunk,
    load_or_compute_q0_action_statistics,
    normalize_action_chunk,
)


class EVOQ0AnchoredDataset(ALOHADataset):
    """Construct 50-step actions from future measured states at sample time.

    HDF5 ``action`` is deliberately only a per-frame storage source and must be
    identical to raw ``observations/qpos``. It is never interpreted as a robot
    command. This subclass replaces the normal contiguous action slicing with
    fixed-anchor chunks and normalizes those resulting deltas with their own
    statistics.
    """

    def __init__(self, *args, **kwargs):
        requested_normalize_actions = bool(kwargs.pop("normalize_actions", True))
        requested_normalize_proprio = bool(kwargs.pop("normalize_proprio", True))
        configured_data_dir = kwargs.get("data_dir", args[0] if args else None)
        statistics_dir = Path(kwargs.pop("statistics_dir", configured_data_dir))
        chunk_size = int(kwargs.get("chunk_size", CHUNK_SIZE))
        if chunk_size != CHUNK_SIZE:
            raise ValueError(f"{CONTRACT_NAME} requires chunk_size={CHUNK_SIZE}, got {chunk_size}")

        # Delay normalization until canonical statistics have been selected.
        # Stage one and stage two may use different episode subsets but must
        # retain exactly one numerical input/output contract.
        super().__init__(*args, normalize_actions=False, normalize_proprio=False, **kwargs)
        self.normalize_q0_actions = requested_normalize_actions
        self.normalize_q0_proprio = requested_normalize_proprio
        self.statistics_dir = statistics_dir

        trajectories = []
        for episode in self.data.values():
            source = episode["actions"]
            if source.ndim != 2 or source.shape[1] != ACTION_DIM:
                raise ValueError(f"expected 17-D measured action source, got {source.shape}")
            if episode["proprio"].shape[1] != PROPRIO_DIM:
                raise ValueError(f"expected 17-D proprio, got {episode['proprio'].shape}")
            self._validate_hdf5_contract(episode["file_path"])
            trajectories.append(source)

        if statistics_dir.resolve() == Path(self.data_dir).resolve():
            q0_stats = load_or_compute_q0_action_statistics(
                self.data_dir,
                trajectories,
                chunk_size=self.chunk_size,
            )
            self.dataset_stats.update(q0_stats)
            self._write_inference_statistics(statistics_dir)
        else:
            statistics_path = statistics_dir / "dataset_statistics.json"
            if not statistics_path.is_file():
                raise FileNotFoundError(
                    f"canonical statistics do not exist: {statistics_path}; "
                    "run projects/evo_q0_towel/compute_statistics.py first"
                )
            self.dataset_stats = {
                key: np.asarray(value, dtype=np.float32)
                for key, value in json.loads(statistics_path.read_text()).items()
            }

        for prefix in ("actions", "proprio"):
            for suffix in ("min", "max", "mean", "std", "median"):
                value = np.asarray(self.dataset_stats[f"{prefix}_{suffix}"])
                if value.shape != (ACTION_DIM,):
                    raise ValueError(f"canonical {prefix}_{suffix} must have 17 values, got {value.shape}")

        if self.normalize_q0_proprio:
            self.data = rescale_data(self.data, self.dataset_stats, "proprio")
            self.rollout_data = rescale_data(self.rollout_data, self.dataset_stats, "proprio")

    @staticmethod
    def _validate_hdf5_contract(path: str) -> None:
        with h5py.File(path, "r") as handle:
            contract = handle.attrs.get("action_contract", "")
            if isinstance(contract, bytes):
                contract = contract.decode()
            if contract != CONTRACT_NAME:
                raise ValueError(f"{path}: expected action_contract={CONTRACT_NAME!r}, got {contract!r}")
            action_source = handle["action"]
            proprio = handle["observations/qpos"]
            if action_source.shape != proprio.shape or action_source.shape[1] != ACTION_DIM:
                raise ValueError(f"{path}: action source and measured proprio must both be [T,17]")
            # Compare in bounded blocks so validation does not duplicate a long
            # trajectory in memory.
            for start in range(0, len(action_source), 4096):
                stop = min(start + 4096, len(action_source))
                if not np.array_equal(action_source[start:stop], proprio[start:stop]):
                    raise ValueError(f"{path}: action source is not identical to measured proprio")

    def _write_inference_statistics(self, statistics_dir: Path) -> None:
        """Write the standard file consumed by Cosmos inference."""
        path = statistics_dir / "dataset_statistics.json"
        payload = {key: np.asarray(value).tolist() for key, value in self.dataset_stats.items()}
        path.write_text(json.dumps(payload, indent=2) + "\n")

    def _get_action_chunk(self, episode_data: dict, relative_step_idx: int) -> np.ndarray:
        chunk = build_q0_anchored_chunk(
            measured_states=episode_data["actions"],
            anchor_index=relative_step_idx,
            chunk_size=self.chunk_size,
        )
        if self.normalize_q0_actions:
            chunk = normalize_action_chunk(chunk, self.dataset_stats)
        return chunk

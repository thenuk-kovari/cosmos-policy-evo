"""LeHome image dataset with measured-state q0-anchored targets.

The prepared Zarr store contains one contiguous timeline per episode.  A sample
at anchor ``i`` has observations from ``i-1, i`` and predicts:

    state[i + 1 : i + 1 + horizon] - state[i]

``action`` from the source dataset is intentionally never read.  This makes
the label a future *observed* joint/gripper state trajectory, anchored at the
query-time proprioception q0.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
import zarr
from threadpoolctl import threadpool_limits

from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


class LeHomeQ0ImageDataset(BaseImageDataset):
    """Fixed-manifest LeHome dataset for a q0 delta diffusion policy."""

    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        split_manifest: str,
        split: str = "train",
        horizon: int = 50,
        n_obs_steps: int = 2,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError(f"split must be train or val, got {split!r}")
        self.shape_meta = shape_meta
        self.dataset_path = str(dataset_path)
        self.split_manifest = str(split_manifest)
        self.split = split
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        if self.n_obs_steps != 2:
            raise ValueError("this baseline intentionally uses the paper's two observation frames")

        self.rgb_keys = [
            key for key, spec in shape_meta["obs"].items() if spec.get("type") == "rgb"
        ]
        self.lowdim_keys = [
            key for key, spec in shape_meta["obs"].items() if spec.get("type", "low_dim") == "low_dim"
        ]
        if self.lowdim_keys != ["q0_state"]:
            raise ValueError("LeHome baseline expects exactly one low-dimensional input: q0_state")

        self._root = None
        root = self._open_root()
        self._episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
        self._episode_ids = list(root.attrs["episode_ids"])
        with open(self.split_manifest) as f:
            manifest = json.load(f)
        expected = set(manifest[split])
        missing = expected.difference(self._episode_ids)
        if missing:
            raise ValueError(f"manifest references episodes absent from Zarr: {sorted(missing)[:5]}")
        self._episode_indices = [
            idx for idx, episode_id in enumerate(self._episode_ids) if episode_id in expected
        ]
        self._anchors = self._build_anchors(self._episode_indices)
        if not len(self._anchors):
            raise ValueError("no valid q0 anchors; episodes must have at least horizon + 2 frames")

    def _open_root(self):
        if self._root is None:
            self._root = zarr.open(self.dataset_path, mode="r")
        return self._root

    def _build_anchors(self, episode_indices: Iterable[int]) -> np.ndarray:
        anchors = []
        previous_end = 0
        for episode_idx, episode_end in enumerate(self._episode_ends):
            if episode_idx in episode_indices:
                # Need i-1, i and future rows i+1 ... i+horizon.
                anchors.extend(range(previous_end + 1, int(episode_end) - self.horizon))
            previous_end = int(episode_end)
        return np.asarray(anchors, dtype=np.int64)

    def get_validation_dataset(self) -> "LeHomeQ0ImageDataset":
        return LeHomeQ0ImageDataset(
            shape_meta=self.shape_meta,
            dataset_path=self.dataset_path,
            split_manifest=self.split_manifest,
            split="val",
            horizon=self.horizon,
            n_obs_steps=self.n_obs_steps,
        )

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        """Fit min/max [-1, 1] normalizers on training anchors only.

        This deliberately mirrors Diffusion Policy's ``limits`` normalizer.
        Validation episodes never affect any bound.
        """
        if self.split != "train":
            raise RuntimeError("normalization must be fit from the training split")
        root = self._open_root()
        state = np.asarray(root["state"][:], dtype=np.float32)
        q0 = state[self._anchors]
        # 58k anchors x 50 x 12 is about 140 MiB for this task: small enough
        # to construct once and ensures bounds describe the actual labels.
        future = np.stack(
            [state[i + 1 : i + 1 + self.horizon] for i in self._anchors], axis=0
        )
        deltas = future - q0[:, None, :]

        normalizer = LinearNormalizer()
        normalizer["q0_state"] = SingleFieldLinearNormalizer.create_fit(q0)
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(deltas)
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        root = self._open_root()
        state = np.asarray(root["state"][:], dtype=np.float32)
        return torch.from_numpy(
            np.stack(
                [state[i + 1 : i + 1 + self.horizon] - state[i] for i in self._anchors], axis=0
            )
        )

    def __len__(self) -> int:
        return len(self._anchors)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        root = self._open_root()
        i = int(self._anchors[index])
        state = np.asarray(root["state"][i - 1 : i + 1], dtype=np.float32)
        q0 = state[-1]
        obs = {"q0_state": state}
        for key in self.rgb_keys:
            # Zarr uses HWC uint8; Diffusion Policy consumes TCHW float [0,1].
            frames = np.asarray(root[f"images/{key}"][i - 1 : i + 1], dtype=np.float32)
            obs[key] = np.moveaxis(frames, -1, 1) / 255.0
        action = np.asarray(root["state"][i + 1 : i + 1 + self.horizon], dtype=np.float32) - q0
        return {
            "obs": {key: torch.from_numpy(value) for key, value in obs.items()},
            "action": torch.from_numpy(action),
        }

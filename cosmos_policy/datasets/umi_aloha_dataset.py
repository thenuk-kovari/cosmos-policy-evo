"""ALOHA-compatible dataset with a fixed 29-D cross-embodiment action contract."""

from __future__ import annotations

import numpy as np

from cosmos_policy.datasets.aloha_dataset import ALOHADataset


class UMIALOHADataset(ALOHADataset):
    """UMI stage: supervise only the bimanual 2x6D EE block.

    Layout: [left joints7, left gripper, right joints7, right gripper,
    elevator, left EE6, right EE6]. Missing embodiment-specific channels are
    present as zeros for checkpoint shape compatibility and excluded from loss.
    """

    ACTION_DIM = 29
    EE_START = 17

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for episode in self.data.values():
            if episode["actions"].shape[1] != self.ACTION_DIM:
                raise ValueError(f"Expected 29-D actions, got {episode['actions'].shape}")
            if episode["proprio"].shape[1] != self.ACTION_DIM:
                raise ValueError(f"Expected 29-D proprio, got {episode['proprio'].shape}")

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        for key in ("actions", "next_action_chunk", "proprio", "future_proprio"):
            sample[key] = np.asarray(sample[key], dtype=np.float32)
        action_dim_mask = np.zeros(self.ACTION_DIM, dtype=np.float32)
        action_dim_mask[self.EE_START :] = 1.0
        sample["action_dim_mask"] = action_dim_mask
        sample["proprio_dim_mask"] = action_dim_mask.copy()
        return sample

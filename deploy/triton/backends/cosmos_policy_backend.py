"""Cosmos Policy 29D/14D adapter for the Evo Triton Python backend."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from backends.base_backend import BaseBackend

ACTION_DIM = 29
PROPRIO_DIM = 14
CHUNK_SIZE = 50


def _install_thor_import_shims(tokenizer_path: Path) -> None:
    import torch

    torch.compile = lambda fn=None, *args, **kwargs: fn if fn is not None else (lambda f: f)

    # The exact final DCP checkpoint must be strict even though the upstream
    # loader permits partial model loads for base-model initialization.
    import cosmos_policy._src.predict2.utils.model_loader as model_loader

    original_planner = model_loader.DefaultLoadPlanner

    def strict_planner(*args, **kwargs):
        kwargs["allow_partial_load"] = False
        return original_planner(*args, **kwargs)

    model_loader.DefaultLoadPlanner = strict_planner

    # The committed experiment graph references the original HF initialization
    # checkpoint while being imported. Final DCP inference does not need it.
    from cosmos_policy._src.imaginaire.utils import checkpoint_db

    checkpoint_db.get_checkpoint_path = lambda path: path

    # The policy DCP excludes the frozen Cosmos video tokenizer. Resolve that
    # one gated base-model asset from the credential-free runtime bundle.
    from cosmos_policy.utils import checkpoint_utils

    original_resolver = checkpoint_utils.resolve_checkpoint_path

    def local_checkpoint_resolver(path):
        if str(path).endswith("tokenizer/tokenizer.pth"):
            return str(tokenizer_path)
        return original_resolver(path)

    checkpoint_utils.resolve_checkpoint_path = local_checkpoint_resolver


def _apply_thor_model_fallbacks(model) -> dict[str, int]:
    """Replace TransformerEngine kernels unavailable on Jetson Thor."""
    import torch
    from cosmos_policy._src.predict2.networks.minimal_v4_dit import RMSNorm as TorchRMSNorm
    from cosmos_policy._src.predict2.networks.minimal_v4_dit import torch_attention_op

    replaced_rmsnorm = 0
    switched_attention = 0

    def replace_children(parent):
        nonlocal replaced_rmsnorm, switched_attention
        for name, child in list(parent.named_children()):
            cls = child.__class__
            if cls.__name__ == "RMSNorm" and cls.__module__.startswith("transformer_engine"):
                dim = int(child.weight.numel())
                eps = float(getattr(child, "eps", 1e-6))
                replacement = TorchRMSNorm(dim, eps=eps).to(
                    device=child.weight.device, dtype=child.weight.dtype
                )
                with torch.no_grad():
                    replacement.weight.copy_(child.weight.detach())
                setattr(parent, name, replacement)
                replaced_rmsnorm += 1
                continue
            if getattr(child, "backend", None) == "transformer_engine" and hasattr(child, "attn_op"):
                child.backend = "torch"
                child.attn_op = torch_attention_op
                if not hasattr(child.attn_op, "set_context_parallel_group"):
                    child.attn_op.set_context_parallel_group = lambda *args, **kwargs: None
                switched_attention += 1
            replace_children(child)

    replace_children(model)
    return {
        "replaced_rmsnorm": replaced_rmsnorm,
        "switched_attention": switched_attention,
    }


def _resolve_model_dir(path: Path) -> Path:
    candidates = (path / "model", path)
    for candidate in candidates:
        if (candidate / ".metadata").exists():
            return candidate
    raise FileNotFoundError(f"No DCP model/.metadata found under {path}")


def _sidecar(params: dict[str, str], key: str, checkpoint_root: Path, filename: str) -> Path:
    configured = params.get(key, "").strip()
    path = Path(configured) if configured else checkpoint_root / filename
    if not path.exists():
        raise FileNotFoundError(f"Required Cosmos sidecar does not exist: {path}")
    return path


def _chw_float_to_hwc_uint8(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[0] != 3:
        raise ValueError(f"{name} must have shape [3,H,W], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8).transpose(1, 2, 0)


class CosmosPolicyBackend(BaseBackend):
    def load_model(self, params: dict[str, str]) -> None:
        os.environ.setdefault("COSMOS_POLICY_SKIP_CONFIG_CHECKPOINT_DOWNLOAD", "1")
        if not any("evo" in arg.lower() or "umi" in arg.lower() for arg in sys.argv):
            sys.argv.append("evo")

        checkpoint_root = Path(self.resolve_checkpoint_path(params))
        model_dir = _resolve_model_dir(checkpoint_root)
        stats_path = _sidecar(
            params, "DATASET_STATS_PATH", checkpoint_root, "dataset_statistics.json"
        )
        t5_path = _sidecar(params, "T5_EMBEDDINGS_PATH", checkpoint_root, "t5_embeddings.pkl")
        tokenizer_path = _sidecar(
            params, "TOKENIZER_PATH", checkpoint_root, "tokenizer.pth"
        )
        task = params.get("TASK_DESCRIPTION", "fold the large blue towel twice")
        experiment = params.get(
            "EXPERIMENT", "predict2-2b-48demos-umi-ee12-8k"
        )
        denoising_steps = int(params.get("DENOISING_STEPS", "10"))

        # Config imports must happen only after the EVO platform selector and
        # local-checkpoint shims are active.
        _install_thor_import_shims(tokenizer_path)

        from cosmos_policy.constants import (
            ACTION_DIM as source_action_dim,
            NUM_ACTIONS_CHUNK as source_chunk_size,
            PROPRIO_DIM as source_proprio_dim,
        )
        from cosmos_policy.experiments.robot.aloha.deploy import DeployConfig
        from cosmos_policy.experiments.robot.cosmos_utils import (
            get_action,
            get_model,
            init_t5_text_embeddings_cache,
            load_dataset_stats,
        )

        actual_contract = (source_action_dim, source_proprio_dim, source_chunk_size)
        expected_contract = (ACTION_DIM, PROPRIO_DIM, CHUNK_SIZE)
        if actual_contract != expected_contract:
            raise RuntimeError(
                f"Cosmos source contract {actual_contract} does not match {expected_contract}"
            )

        self.cfg = DeployConfig(
            suite="aloha",
            config=experiment,
            ckpt_path=str(model_dir),
            config_file="cosmos_policy/config/config.py",
            use_third_person_image=True,
            num_third_person_images=1,
            use_wrist_image=True,
            num_wrist_images=2,
            use_proprio=True,
            normalize_proprio=True,
            unnormalize_actions=True,
            dataset_stats_path=str(stats_path),
            t5_text_embeddings_path=str(t5_path),
            trained_with_image_aug=True,
            chunk_size=CHUNK_SIZE,
            num_open_loop_steps=CHUNK_SIZE,
            ar_future_prediction=False,
            ar_value_prediction=False,
            ar_qvalue_prediction=False,
            use_jpeg_compression=False,
            flip_images=False,
            num_denoising_steps_action=denoising_steps,
            deterministic=True,
            seed=195,
            randomize_seed=False,
            num_queries_best_of_n=1,
            use_parallel_inference=False,
        )
        init_t5_text_embeddings_cache(str(t5_path))
        self.dataset_stats = load_dataset_stats(str(stats_path))
        self.model, self.cosmos_config = get_model(self.cfg)
        self.fallbacks = _apply_thor_model_fallbacks(self.model)
        self.task = task
        self._get_action = get_action

        train_chunk = self.cosmos_config.dataloader_train.dataset.chunk_size
        if train_chunk != CHUNK_SIZE:
            raise RuntimeError(f"Checkpoint config chunk size is {train_chunk}, expected {CHUNK_SIZE}")

        if np.asarray(self.dataset_stats["actions_min"]).shape != (ACTION_DIM,):
            raise RuntimeError("dataset statistics do not contain 29 action dimensions")
        if np.asarray(self.dataset_stats["proprio_min"]).shape != (PROPRIO_DIM,):
            raise RuntimeError("dataset statistics do not contain 14 proprio dimensions")

    def preprocess(self, raw_inputs: dict[str, np.ndarray]) -> dict:
        required = (
            "observation__state",
            "observation__images__base",
            "observation__images__left_gripper",
            "observation__images__right_gripper",
        )
        missing = [name for name in required if name not in raw_inputs]
        if missing:
            raise ValueError(f"Missing required Cosmos inputs: {missing}")

        proprio = np.asarray(raw_inputs["observation__state"], dtype=np.float32).reshape(-1)
        if proprio.shape != (PROPRIO_DIM,):
            raise ValueError(f"observation__state must have shape [14], got {proprio.shape}")
        if not np.isfinite(proprio).all():
            raise ValueError("observation__state contains NaN or infinity")

        return {
            "observation": {
                "task_description": self.task,
                "proprio": proprio,
                "primary_image": _chw_float_to_hwc_uint8(
                    raw_inputs["observation__images__base"], "base image"
                ),
                "left_wrist_image": _chw_float_to_hwc_uint8(
                    raw_inputs["observation__images__left_gripper"], "left wrist image"
                ),
                "right_wrist_image": _chw_float_to_hwc_uint8(
                    raw_inputs["observation__images__right_gripper"], "right wrist image"
                ),
            }
        }

    def infer(self, processed_inputs: dict) -> dict:
        result = self._get_action(
            self.cfg,
            self.model,
            self.dataset_stats,
            processed_inputs["observation"],
            self.task,
            seed=self.cfg.seed,
            randomize_seed=False,
            num_denoising_steps_action=self.cfg.num_denoising_steps_action,
            generate_future_state_and_value_in_parallel=False,
        )
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.shape != (CHUNK_SIZE, ACTION_DIM):
            raise RuntimeError(f"Cosmos returned {actions.shape}, expected [50,29]")
        if not np.isfinite(actions).all():
            raise RuntimeError("Cosmos returned non-finite actions")
        return {"action": actions}

    def postprocess(self, raw_outputs: dict) -> dict[str, np.ndarray]:
        return {"action": raw_outputs["action"]}

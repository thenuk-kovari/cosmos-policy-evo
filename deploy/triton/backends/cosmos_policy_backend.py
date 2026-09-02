"""Cosmos Policy adapter for the Evo Triton Python backend.

Supports native 17D Evo q0 heads and 35D EE+joint transfer heads while always
exposing the established 17D q0 runtime contract to the Orin.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from backends.base_backend import BaseBackend

DEPLOYED_ACTION_DIM = 17
PROPRIO_DIM = 17
CHUNK_SIZE = 50

_SOURCE_CONTRACTS = {
    "evo_q0_observed_state_left_first_v1": {
        "source_action_dim": 17,
        "experiment": "predict2-2b-evo-q0-state17",
        "platform_argument": "evo-q0",
        "executable_action_slice": slice(0, 17),
    },
    "evo_ee6d_joint35_left_first_q0_v1": {
        "source_action_dim": 35,
        "experiment": "predict2-2b-evo-ee6d-joint35-teleop",
        "platform_argument": "genrobot-ee6d",
        "executable_action_slice": slice(18, 35),
    },
}


def _runtime_contract(checkpoint_root: Path) -> dict:
    """Resolve the checkpoint's native head from its required sidecar."""
    config_path = checkpoint_root / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Required policy sidecar does not exist: {config_path}")
    with config_path.open() as stream:
        policy = json.load(stream)
    source_contract = policy.get("source_contract", policy.get("contract"))
    if source_contract not in _SOURCE_CONTRACTS:
        raise RuntimeError(f"Unsupported Cosmos source contract: {source_contract!r}")
    resolved = dict(_SOURCE_CONTRACTS[source_contract])
    expected_shape = [CHUNK_SIZE, resolved["source_action_dim"]]
    declared_shape = policy.get("source_action_shape", expected_shape)
    if declared_shape != expected_shape:
        raise RuntimeError(
            f"Policy declares source_action_shape={declared_shape}, expected "
            f"{expected_shape} for {source_contract}"
        )
    resolved["source_contract"] = source_contract
    resolved["experiment"] = policy.get("experiment", resolved["experiment"])
    resolved["task"] = policy.get("task_description", "fold the blue towel twice")
    return resolved


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


def _load_t5_cache_readonly(path: Path, required_task: str) -> None:
    """Load a precomputed T5 cache without creating lock files beside it."""
    import pickle
    import torch
    import cosmos_policy.experiments.robot.cosmos_utils as cosmos_utils

    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict) or required_task not in data:
        raise RuntimeError(
            f"T5 cache {path} does not contain required task {required_task!r}; "
            f"available keys={list(data) if isinstance(data, dict) else type(data)}"
        )
    embedding = data[required_task]
    if not isinstance(embedding, torch.Tensor) or tuple(embedding.shape[-2:]) != (512, 1024):
        raise RuntimeError(f"Invalid T5 embedding for {required_task!r}: {type(embedding)}, {getattr(embedding, 'shape', None)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cosmos_utils.t5_text_embeddings_cache.clear()
    cosmos_utils.t5_text_embeddings_cache.update(
        {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in data.items()}
    )
    cosmos_utils.t5_text_embeddings_path_global = str(path)
    cosmos_utils.t5_text_embeddings_newly_computed = False


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


def _image_to_hwc_uint8(value: np.ndarray, name: str) -> np.ndarray:
    """Decode fresh-main JPEG transport; retain direct-call CHW test support."""
    array = np.asarray(value)
    if array.ndim == 1 and array.dtype == np.uint8:
        import cv2

        bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"{name} is not a valid JPEG payload")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    if array.ndim != 3 or array.shape[0] != 3:
        raise ValueError(f"{name} must be uint8 JPEG [N] or float CHW [3,H,W], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8).transpose(1, 2, 0)


class CosmosPolicyBackend(BaseBackend):
    def load_model(self, params: dict[str, str]) -> None:
        os.environ.setdefault("COSMOS_POLICY_SKIP_CONFIG_CHECKPOINT_DOWNLOAD", "1")
        checkpoint_root = Path(self.resolve_checkpoint_path(params))
        runtime = _runtime_contract(checkpoint_root)
        if not any(
            token in arg.lower()
            for arg in sys.argv
            for token in ("genrobot", "ee6d", "evo-q0", "evo_q0")
        ):
            sys.argv.append(runtime["platform_argument"])
        model_dir = _resolve_model_dir(checkpoint_root)
        stats_path = _sidecar(
            params, "DATASET_STATS_PATH", checkpoint_root, "dataset_statistics.json"
        )
        t5_path = _sidecar(params, "T5_EMBEDDINGS_PATH", checkpoint_root, "t5_embeddings.pkl")
        tokenizer_path = _sidecar(
            params, "TOKENIZER_PATH", checkpoint_root, "tokenizer.pth"
        )
        task = runtime.get("task") or params.get("TASK_DESCRIPTION", "fold the blue towel twice")
        experiment = runtime.get("experiment") or params.get("EXPERIMENT")
        expected_source_action_dim = int(runtime["source_action_dim"])
        executable_action_slice = runtime["executable_action_slice"]
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
            load_dataset_stats,
        )

        actual_contract = (source_action_dim, source_proprio_dim, source_chunk_size)
        expected_contract = (expected_source_action_dim, PROPRIO_DIM, CHUNK_SIZE)
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
        _load_t5_cache_readonly(t5_path, task)
        self.dataset_stats = load_dataset_stats(str(stats_path))
        self.model, self.cosmos_config = get_model(self.cfg)
        # Cosmos Policy's evaluated Predict2 deployment schedule deliberately
        # differs from the training SDE (200 -> 0.01).  Keep the initial noise
        # deterministic, but sample over the cookbook's inference range.
        self.model.sde.sigma_max = float(params.get("SIGMA_MAX", "80"))
        self.model.sde.sigma_min = float(params.get("SIGMA_MIN", "4"))
        if (self.model.sde.sigma_max, self.model.sde.sigma_min) != (80.0, 4.0):
            raise RuntimeError(
                "Cosmos Predict2 inference requires sigma_max=80 and sigma_min=4"
            )
        self.fallbacks = _apply_thor_model_fallbacks(self.model)
        self.task = task
        self._get_action = get_action
        self.source_action_dim = expected_source_action_dim
        self.executable_action_slice = executable_action_slice

        train_chunk = self.cosmos_config.dataloader_train.dataset.chunk_size
        if train_chunk != CHUNK_SIZE:
            raise RuntimeError(f"Checkpoint config chunk size is {train_chunk}, expected {CHUNK_SIZE}")

        if np.asarray(self.dataset_stats["actions_min"]).shape != (self.source_action_dim,):
            raise RuntimeError(
                f"dataset statistics do not contain {self.source_action_dim} action dimensions"
            )
        if np.asarray(self.dataset_stats["proprio_min"]).shape != (PROPRIO_DIM,):
            raise RuntimeError("dataset statistics do not contain 17 proprio dimensions")

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
            raise ValueError(f"observation__state must have shape [17], got {proprio.shape}")
        if not np.isfinite(proprio).all():
            raise ValueError("observation__state contains NaN or infinity")

        return {
            "observation": {
                "task_description": self.task,
                "proprio": proprio,
                "primary_image": _image_to_hwc_uint8(
                    raw_inputs["observation__images__base"], "base image"
                ),
                "left_wrist_image": _image_to_hwc_uint8(
                    raw_inputs["observation__images__left_gripper"], "left wrist image"
                ),
                "right_wrist_image": _image_to_hwc_uint8(
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
        if actions.shape != (CHUNK_SIZE, self.source_action_dim):
            raise RuntimeError(
                f"Cosmos returned {actions.shape}, expected [50,{self.source_action_dim}]"
            )
        if not np.isfinite(actions).all():
            raise RuntimeError("Cosmos returned non-finite actions")
        deployed_actions = actions[:, self.executable_action_slice]
        if deployed_actions.shape != (CHUNK_SIZE, DEPLOYED_ACTION_DIM):
            raise RuntimeError(
                f"Executable action slice has {deployed_actions.shape}, expected [50,17]"
            )
        return {"action": deployed_actions}

    def postprocess(self, raw_outputs: dict) -> dict[str, np.ndarray]:
        return {"action": raw_outputs["action"]}

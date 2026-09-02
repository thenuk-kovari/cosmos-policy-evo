"""Evaluate one GenRobot checkpoint on the held-out split.

This intentionally calls the policy training objective under ``no_grad``. The
built-in Predict2 ``validation_step`` performs generative sampling and does not
return the action diagnostics logged during policy training, so it cannot
produce a comparable action-loss curve.
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_policy._src.imaginaire.config import load_config
from cosmos_policy._src.imaginaire.lazy_config import instantiate
from cosmos_policy._src.imaginaire.utils import distributed, misc
from cosmos_policy._src.imaginaire.utils.context_managers import data_loader_init, distributed_init, model_init
from cosmos_policy._src.predict2.utils.model_loader import create_model_from_consolidated_checkpoint_with_fsdp


METRICS = (
    "demo_sample_action_mse_loss",
    "demo_sample_action_l1_loss",
    "demo_sample_future_proprio_mse_loss",
    "world_model_sample_future_proprio_mse_loss",
    "edm_loss",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    opts = [
        "--",
        "experiment=predict2-2b-genrobot-towel-ee6d35",
        f"checkpoint.load_path={args.checkpoint}",
        "checkpoint.load_training_state=false",
        "checkpoint.load_from_object_store.enabled=true",
        "checkpoint.save_to_object_store.enabled=false",
        "job.wandb_mode=disabled",
        "trainer.callbacks.compile_tokenizer.enabled=false",
    ]
    config = load_config("cosmos_policy/config/config.py", opts, enable_one_logger=True)
    with distributed_init():
        distributed.init()
    config.validate()
    # Select held-out data before freezing the otherwise training-identical
    # configuration.
    config.dataloader_train.dataset.is_train = False
    config.dataloader_train.dataset.use_image_aug = False
    config.dataloader_train.dataset.use_stronger_image_aug = False
    config.freeze()
    trainer = config.trainer.type(config)

    torch.manual_seed(20260813)
    np.random.seed(20260813)
    random.seed(20260813)
    with model_init():
        checkpoint_path = str(config.checkpoint.load_path or "")
        if checkpoint_path.endswith(".pt"):
            model = create_model_from_consolidated_checkpoint_with_fsdp(config)
        else:
            model = instantiate(config.model)

    with data_loader_init():
        # Clone the configured training dataset node, changing only the split.
        dataset = instantiate(config.dataloader_train.dataset)
        sampler = DistributedSampler(
            dataset,
            num_replicas=parallel_state.get_data_parallel_world_size(),
            rank=parallel_state.get_data_parallel_rank(),
            shuffle=False,
            seed=20260813,
            drop_last=False,
        )
        loader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=config.dataloader_train.batch_size,
            drop_last=False,
            num_workers=config.dataloader_train.num_workers,
            persistent_workers=config.dataloader_train.persistent_workers,
            pin_memory=config.dataloader_train.pin_memory,
        )

    model = model.to("cuda", memory_format=config.trainer.memory_format)
    model.on_train_start(config.trainer.memory_format)
    trainer.checkpointer.load(model)
    # ``training_step`` is the only path that emits the action diagnostic. Its
    # network path also performs the configured BF16 casts in training mode.
    # Gradients remain disabled below, and the RNG is fixed for comparability.
    model.train()

    totals = {name: torch.zeros((), device="cuda", dtype=torch.float64) for name in METRICS}
    counts = {name: torch.zeros((), device="cuda", dtype=torch.float64) for name in METRICS}
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= args.max_batches:
                break
            batch = misc.to(batch, device="cuda")
            # Mirror LowPrecisionCallback.on_training_step_start from the
            # production trainer. Without this, FP32 observations meet BF16
            # DiT weights and the comparison is invalid.
            for key, value in batch.items():
                if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                    batch[key] = value.to(dtype=model.precision)
            output, _ = model.training_step(batch, args.step)
            for name in METRICS:
                value = output[name].detach().double()
                if torch.isfinite(value):
                    totals[name] += value
                    counts[name] += 1

    result = {"step": args.step, "split": "val", "max_batches_per_rank": args.max_batches}
    for name in METRICS:
        dist.all_reduce(totals[name], op=dist.ReduceOp.SUM)
        dist.all_reduce(counts[name], op=dist.ReduceOp.SUM)
        result[f"val/{name}"] = (totals[name] / counts[name].clamp_min(1)).item()
        result[f"val/{name}_count"] = int(counts[name].item())

    if distributed.is_rank0():
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
        print(json.dumps(result, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

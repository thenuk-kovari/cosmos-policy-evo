#!/usr/bin/env bash
set -euo pipefail

: "${YAM_FOLD_TOWEL_DATA_DIR:?Set YAM_FOLD_TOWEL_DATA_DIR to the local dataset root}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY through the runtime secret environment}"

# W&B's device-monitor tables use the artifact staging area after a few steps.
# Keep all W&B writes on the durable, container-writable output volume.
if [[ -n "${IMAGINAIRE_OUTPUT_ROOT:-}" ]]; then
    export WANDB_DATA_DIR="${WANDB_DATA_DIR:-${IMAGINAIRE_OUTPUT_ROOT}/.wandb-data}"
    export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${IMAGINAIRE_OUTPUT_ROOT}/.wandb-cache}"
    mkdir -p "${WANDB_DATA_DIR}" "${WANDB_CACHE_DIR}"
fi

EXPERIMENT="predict2-2b-23demos-75policy-25world"
CONFIG="cosmos_policy/config/config.py"
EXTRA_OVERRIDES=()

if [[ "${RESUME_FROM_S3:-0}" == "1" ]]; then
    EXTRA_OVERRIDES+=(
        "checkpoint.load_from_object_store.enabled=true"
        "checkpoint.load_from_object_store.bucket=policy-training"
    )
fi

uv run --extra cu128 --group aloha --python 3.10 \
    torchrun \
    --nproc_per_node=8 \
    --master_port="${MASTER_PORT:-12341}" \
    -m cosmos_policy.scripts.train \
    --config="${CONFIG}" -- \
    experiment="${EXPERIMENT}" \
    trainer.callbacks.compile_tokenizer.enabled=false \
    "${EXTRA_OVERRIDES[@]}"

#!/usr/bin/env bash
set -euo pipefail

: "${UMI_LARGE_BLUE_TOWEL_DATA_DIR:?Set UMI_LARGE_BLUE_TOWEL_DATA_DIR}"

export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-/home/thenuk-kovari/cosmos-policy-output}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-${IMAGINAIRE_OUTPUT_ROOT}/.wandb-data}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${IMAGINAIRE_OUTPUT_ROOT}/.wandb-cache}"
export YAM_S3_CREDENTIALS="${YAM_S3_CREDENTIALS:-/home/thenuk-kovari/.secrets/cosmos_s3.json}"
mkdir -p "${WANDB_DATA_DIR}" "${WANDB_CACHE_DIR}"

EXPERIMENT="predict2-2b-48demos-umi-ee12-8k"
CONFIG="cosmos_policy/config/config.py"
MAX_ITER="${MAX_ITER:-8000}"
SAVE_ITER="${SAVE_ITER:-1000}"
RUN_NAME="${RUN_NAME:-${EXPERIMENT}}"
SAVE_TO_S3="${SAVE_TO_S3:-true}"
WANDB_MODE="${WANDB_MODE:-online}"
UV_BIN="${UV_BIN:-/home/thenuk-kovari/cosmos-policy-venv/bin/uv}"

"${UV_BIN}" run --extra cu128 --group aloha --python 3.10 \
    torchrun \
    --nproc_per_node=8 \
    --master_port="${MASTER_PORT:-12342}" \
    -m cosmos_policy.scripts.train \
    --config="${CONFIG}" -- \
    experiment="${EXPERIMENT}" \
    trainer.max_iter="${MAX_ITER}" \
    checkpoint.save_iter="${SAVE_ITER}" \
    job.name="${RUN_NAME}" \
    job.wandb_mode="${WANDB_MODE}" \
    checkpoint.save_to_object_store.enabled="${SAVE_TO_S3}" \
    checkpoint.save_to_object_store.credentials="${YAM_S3_CREDENTIALS}" \
    trainer.callbacks.compile_tokenizer.enabled=false

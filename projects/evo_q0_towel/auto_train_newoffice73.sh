#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ubuntu/video-policy}"
DATA="${DATA:-/home/ubuntu/data/evo_q0_newoffice73}"
OUTPUT="${OUTPUT:-/home/ubuntu/cosmos-output}"
LOG_DIR="${LOG_DIR:-/home/ubuntu/logs}"
UV="${UV:-/home/ubuntu/.local/bin/uv}"
PREP_LOG="${PREP_LOG:-${LOG_DIR}/prepare-new73.log}"

while tmux has-session -t prepare-new73 2>/dev/null; do
  sleep 10
done
grep -q 'PREPARATION_COMPLETE' "${PREP_LOG}"

mkdir -p "${OUTPUT}" "${LOG_DIR}" "${OUTPUT}/.wandb-data" "${OUTPUT}/.wandb-cache"
export EVO_Q0_DATA_DIR="${DATA}"
export EVO_Q0_STATS_DIR="${DATA}/train_statistics"
export IMAGINAIRE_OUTPUT_ROOT="${OUTPUT}"
export COSMOS_S3_CREDENTIALS=/home/ubuntu/.secrets/cosmos_s3.json
export WANDB_DATA_DIR="${OUTPUT}/.wandb-data"
export WANDB_CACHE_DIR="${OUTPUT}/.wandb-cache"
export HF_TOKEN="$(< /home/ubuntu/.cache/huggingface/token)"

# Exercise model initialization plus distributed forward/backward before the
# paid full run. No online logging or checkpoint upload occurs for this smoke.
"${UV}" run --extra cu128 --group aloha --python 3.10 \
  torchrun --nproc_per_node=8 --master_port=12445 \
  -m cosmos_policy.scripts.train --config=cosmos_policy/config/config.py -- \
  experiment=predict2-2b-evo-q0-state17 \
  trainer.max_iter=2 \
  checkpoint.save_iter=999999 \
  job.name=evo-q0-newoffice73-smoke \
  job.wandb_mode=disabled \
  checkpoint.save_to_object_store.enabled=false \
  trainer.callbacks.compile_tokenizer.enabled=false \
  > "${LOG_DIR}/train-smoke.log" 2>&1

export JOB_NAME=evo-q0-newoffice73-8k
export MAX_STEPS=8000
export SAVE_EVERY=500
exec ./projects/evo_q0_towel/train.sh > "${LOG_DIR}/train-full.log" 2>&1

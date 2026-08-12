#!/usr/bin/env bash
set -euo pipefail

: "${GENROBOT_EE_Q0_DATA_DIR:?Set GENROBOT_EE_Q0_DATA_DIR to the converted dataset root}"
: "${IMAGINAIRE_OUTPUT_ROOT:?Set IMAGINAIRE_OUTPUT_ROOT to persistent output storage}"

MAX_STEPS="${MAX_STEPS:-12000}"
SAVE_EVERY="${SAVE_EVERY:-500}"
JOB_NAME="${JOB_NAME:-genrobot-towel-ee6d35-${MAX_STEPS}}"
MASTER_PORT="${MASTER_PORT:-12447}"
UV="${UV:-$HOME/.local/bin/uv}"

args=(
  --config=cosmos_policy/config/config.py
  --
  experiment=predict2-2b-genrobot-towel-ee6d35
  "trainer.max_iter=${MAX_STEPS}"
  "checkpoint.save_iter=${SAVE_EVERY}"
  "job.name=${JOB_NAME}"
  job.wandb_mode=online
  checkpoint.save_to_object_store.enabled=true
  trainer.callbacks.compile_tokenizer.enabled=false
)

if [[ -n "${LOAD_PATH:-}" ]]; then
  args+=(
    checkpoint.load_from_object_store.enabled=true
    "checkpoint.load_path=${LOAD_PATH}"
    "checkpoint.load_training_state=${LOAD_TRAINING_STATE:-false}"
  )
fi

exec "$UV" run --extra cu128 --group aloha --python 3.10 \
  torchrun --nproc_per_node=8 --master_port="$MASTER_PORT" \
  -m cosmos_policy.scripts.train "${args[@]}"

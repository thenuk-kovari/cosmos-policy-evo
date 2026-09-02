#!/usr/bin/env bash
set -euo pipefail

: "${EVO_EE_JOINT_Q0_DATA_DIR:?Set EVO_EE_JOINT_Q0_DATA_DIR to the converted Evo dataset}"
: "${IMAGINAIRE_OUTPUT_ROOT:?Set IMAGINAIRE_OUTPUT_ROOT to persistent output storage}"
: "${LOAD_PATH:?Set LOAD_PATH to the stage-one GenRobot/UMI checkpoint URI}"

MAX_STEPS="${MAX_STEPS:-12000}"
SAVE_EVERY="${SAVE_EVERY:-500}"
JOB_NAME="${JOB_NAME:-evo-ee6d-joint35-teleop-${MAX_STEPS}}"
MASTER_PORT="${MASTER_PORT:-12457}"
UV="${UV:-$HOME/.local/bin/uv}"

exec "$UV" run --extra cu128 --group aloha --python 3.10 \
  torchrun --nproc_per_node=8 --master_port="$MASTER_PORT" \
  -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py \
  -- \
  experiment=predict2-2b-evo-ee6d-joint35-teleop \
  "trainer.max_iter=${MAX_STEPS}" \
  "checkpoint.save_iter=${SAVE_EVERY}" \
  "job.name=${JOB_NAME}" \
  job.wandb_mode=online \
  checkpoint.load_from_object_store.enabled=true \
  "checkpoint.load_path=${LOAD_PATH}" \
  checkpoint.load_training_state=false \
  checkpoint.save_to_object_store.enabled=true \
  trainer.callbacks.compile_tokenizer.enabled=false

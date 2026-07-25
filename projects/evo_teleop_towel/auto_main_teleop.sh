#!/usr/bin/env bash
set -euo pipefail
REPO=/home/thenuk-kovari/cosmos-policy
OUTPUT=/home/thenuk-kovari/cosmos-policy-output
DATA=/home/thenuk-kovari/data/evo_teleop_cosmos_25hz
CREDS=/home/thenuk-kovari/.secrets/cosmos_s3.json
UMI_LOG="$OUTPUT/logs/umi-v2-action-2500-resume.log"
TRAIN_LOG="$OUTPUT/logs/umi-then-teleop-action-8k.log"
PIPELINE_LOG="$OUTPUT/logs/umi-then-teleop-pipeline.log"
log(){ printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$PIPELINE_LOG"; }
log 'waiting for UMI action stage'
until grep -q 'Saved checkpoint to s3://policy-training/.*/iter_000010000' "$UMI_LOG"; do
  if ! pgrep -x pt_elastic >/dev/null; then
    log 'ABORT UMI action process exited before verified iter_000010000'
    exit 1
  fi
  sleep 30
done
log 'UMI checkpoint verified; waiting for its GPU workers to exit'
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" ]]; do sleep 10; done
export AWS_ACCESS_KEY_ID="$(python3 -c 'import json;print(json.load(open("/home/thenuk-kovari/.secrets/cosmos_s3.json"))["aws_access_key_id"])')"
export AWS_SECRET_ACCESS_KEY="$(python3 -c 'import json;print(json.load(open("/home/thenuk-kovari/.secrets/cosmos_s3.json"))["aws_secret_access_key"])')"
S3_ENDPOINT="$(python3 -c 'import json;print(json.load(open("/home/thenuk-kovari/.secrets/cosmos_s3.json"))["endpoint_url"])')"
AWS_REGION="$(python3 -c 'import json;print(json.load(open("/home/thenuk-kovari/.secrets/cosmos_s3.json"))["region_name"])')"
export AWS_DEFAULT_REGION="$AWS_REGION"
log 'UMI checkpoint complete; waiting for processed teleop dataset'
until aws --endpoint-url "$S3_ENDPOINT" s3api head-object --bucket policy-training --key datasets/evo-teleop-25hz/conversion_manifest.json >/dev/null 2>&1; do sleep 30; done
mkdir -p "$DATA"
aws --endpoint-url "$S3_ENDPOINT" s3 sync s3://policy-training/datasets/evo-teleop-25hz/ "$DATA" --only-show-errors
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
python3 - "$DATA" <<'PY'
import json,sys
from pathlib import Path
import h5py,numpy as np
p=Path(sys.argv[1]); fs=sorted((p/'train').glob('episode_*.hdf5')); m=json.loads((p/'conversion_manifest.json').read_text())
assert len(fs)==50 and len(m['episodes'])==50
for f in fs:
 with h5py.File(f,'r') as h:
  assert h['action'].shape[1]==29 and h['observations/qpos'].shape[1]==14 and np.all(h['action_dim_mask'][:]==1)
for name in ['dataset_statistics.json','dataset_statistics_post_norm.json','t5_embeddings.pkl']:
 assert (p/name).is_file(),name
print('teleop dataset validated',len(fs),m['total_frames'])
PY
log 'teleop dataset downloaded and validated; launching fresh 8k action-only adaptation from UMI iter 10000'
cd "$REPO"
env UMI_LARGE_BLUE_TOWEL_DATA_DIR="$DATA" WANDB_API_KEY="$(< /home/thenuk-kovari/.secrets/wandb_api_key)" IMAGINAIRE_OUTPUT_ROOT="$OUTPUT" WANDB_DATA_DIR="$OUTPUT/.wandb-data" WANDB_CACHE_DIR="$OUTPUT/.wandb-cache" /home/thenuk-kovari/cosmos-policy-venv/bin/uv run --extra cu128 --group aloha --python 3.10 torchrun --nproc_per_node=8 --master_port=12445 -m cosmos_policy.scripts.train --config=cosmos_policy/config/config.py -- experiment=predict2-2b-48demos-umi-ee12-8k trainer.max_iter=8000 checkpoint.save_iter=500 job.name=umi-then-teleop-action-8k job.wandb_mode=online checkpoint.save_to_object_store.enabled=true checkpoint.save_to_object_store.credentials="$CREDS" checkpoint.load_from_object_store.enabled=true checkpoint.load_from_object_store.credentials="$CREDS" checkpoint.load_path=cosmos-policy-large-blue-towel/umi-ee12/umi-v2-world-7500/checkpoints/iter_000010000 checkpoint.load_training_state=false trainer.callbacks.compile_tokenizer.enabled=false >> "$TRAIN_LOG" 2>&1
if ! grep -q 'Saved checkpoint to s3://policy-training/.*/iter_000008000' "$TRAIN_LOG"; then log 'ABORT teleop adaptation lacks verified iter_000008000'; exit 1; fi
touch "$OUTPUT/MAIN_UMI_TELEOP_COMPLETE"
log 'main UMI-to-teleop model complete and uploaded; shutting down'
sync
sudo -n shutdown -h now

#!/usr/bin/env bash
set -euo pipefail
REPO=/home/ubuntu/cosmos-policy
DATA=/home/ubuntu/data/evo_teleop_cosmos_25hz
CREDS=/home/ubuntu/.secrets/cosmos_s3.json
PIPELINE_LOG=/home/ubuntu/data/logs/evo_teleop_ablation_pipeline.log
TRAIN_LOG=/home/ubuntu/data/logs/evo_teleop_ablation_train.log
log(){ printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$PIPELINE_LOG"; }
log 'waiting for conversion'
while pgrep -f 'prepare_raw_mcap.py.*evo_teleop_cosmos_25hz' >/dev/null; do sleep 30; done
if [[ ! -f "$DATA/conversion_manifest.json" ]]; then log 'ABORT conversion failed'; exit 1; fi
python3 "$REPO/projects/evo_teleop_towel/finalize_dataset.py" "$DATA" --embedding /home/ubuntu/data/t5_embeddings.pkl >> "$PIPELINE_LOG" 2>&1
export AWS_ACCESS_KEY_ID="$(python3 -c 'import json;print(json.load(open("/home/ubuntu/.secrets/cosmos_s3.json"))["aws_access_key_id"])')"
export AWS_SECRET_ACCESS_KEY="$(python3 -c 'import json;print(json.load(open("/home/ubuntu/.secrets/cosmos_s3.json"))["aws_secret_access_key"])')"
S3_ENDPOINT="$(python3 -c 'import json;print(json.load(open("/home/ubuntu/.secrets/cosmos_s3.json"))["endpoint_url"])')"
AWS_REGION="$(python3 -c 'import json;print(json.load(open("/home/ubuntu/.secrets/cosmos_s3.json"))["region_name"])')"
export AWS_DEFAULT_REGION="$AWS_REGION"
log 'dataset finalized; uploading to s3://policy-training/datasets/evo-teleop-25hz/'
aws --endpoint-url "$S3_ENDPOINT" s3 sync "$DATA" s3://policy-training/datasets/evo-teleop-25hz/ --only-show-errors
aws --endpoint-url "$S3_ENDPOINT" s3api head-object --bucket policy-training --key datasets/evo-teleop-25hz/conversion_manifest.json >/dev/null
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
log 'dataset upload verified; launching teleop-only 25-world/75-action ablation'
cd "$REPO"
env UMI_LARGE_BLUE_TOWEL_DATA_DIR="$DATA" WANDB_API_KEY="$(< /home/ubuntu/.secrets/wandb_api_key)" HF_TOKEN="$(< /home/ubuntu/.cache/huggingface/token)" IMAGINAIRE_OUTPUT_ROOT=/home/ubuntu/cosmos-policy-output WANDB_DATA_DIR=/home/ubuntu/cosmos-policy-output/.wandb-data WANDB_CACHE_DIR=/home/ubuntu/cosmos-policy-output/.wandb-cache /home/ubuntu/.local/bin/uv run --extra cu128 --group aloha --python 3.10 torchrun --nproc_per_node=8 --master_port=12444 -m cosmos_policy.scripts.train --config=cosmos_policy/config/config.py -- experiment=predict2-2b-48demos-umi-ee12-8k trainer.max_iter=10000 checkpoint.save_iter=500 job.name=evo-teleop-ablation-25world-75action-10k job.wandb_mode=online checkpoint.save_to_object_store.enabled=true checkpoint.save_to_object_store.credentials=/home/ubuntu/.secrets/cosmos_s3.json trainer.callbacks.compile_tokenizer.enabled=false >> "$TRAIN_LOG" 2>&1
if ! grep -q 'Saved checkpoint to s3://policy-training/.*/iter_000010000' "$TRAIN_LOG"; then log 'ABORT training exited without verified S3 iter_000010000'; exit 1; fi
touch /home/ubuntu/data/TELEOP_ABLATION_COMPLETE
log 'teleop ablation complete and uploaded; shutting down'
sync
sudo -n shutdown -h now

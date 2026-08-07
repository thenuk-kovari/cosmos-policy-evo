#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ubuntu/video-policy}"
DATA_ROOT="${DATA_ROOT:-/home/ubuntu/data}"
OUTPUT="${OUTPUT:-${DATA_ROOT}/evo_q0_newoffice73}"
UV="${UV:-/home/ubuntu/.local/bin/uv}"

while tmux has-session -t download-new50 2>/dev/null || tmux has-session -t download-new25 2>/dev/null; do
  sleep 10
done

new50_count="$(find "${DATA_ROOT}/new50/data" -name 'file-*.parquet' -type f | wc -l)"
new25_count="$(find "${DATA_ROOT}/new25/data" -name 'file-*.parquet' -type f | wc -l)"
if [[ "${new50_count}" != 49 || "${new25_count}" != 31 ]]; then
  echo "download incomplete: new50=${new50_count}/49 new25=${new25_count}/31" >&2
  exit 1
fi

cd "${ROOT}"
"${UV}" run --extra cu128 --group aloha --python 3.10 \
  python projects/evo_q0_towel/prepare_parquet.py \
  --dataset "new50=${DATA_ROOT}/new50" \
  --dataset "new25=${DATA_ROOT}/new25" \
  --selection projects/evo_q0_towel/accepted_new_office_73.json \
  --val-count 6 \
  --out "${OUTPUT}" \
  --overwrite

"${UV}" run --extra cu128 --group aloha --python 3.10 \
  python projects/evo_q0_towel/compute_statistics.py \
  --data-dir "${OUTPUT}" \
  --out "${OUTPUT}/train_statistics"

cp /home/ubuntu/t5_embeddings_fold_blue_towel_twice.pkl "${OUTPUT}/t5_embeddings.pkl"
"${UV}" run --extra cu128 --group aloha --python 3.10 \
  python projects/evo_q0_towel/validate_dataset.py "${OUTPUT}"

python3 - "${OUTPUT}/conversion_manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
assert manifest["counts"] == {"total": 73, "train": 67, "val": 6}, manifest["counts"]
assert not any(
    row["source_dataset"] == "new50" and row["source_file_index"] == 38
    for row in manifest["episodes"]
)
assert not any(
    row["source_dataset"] == "new25" and row["source_file_index"] in {6, 11, 16, 20, 22, 26}
    for row in manifest["episodes"]
)
print("PREPARATION_COMPLETE", manifest["counts"], manifest["total_frames"])
PY

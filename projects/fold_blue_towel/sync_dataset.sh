#!/usr/bin/env bash
set -euo pipefail

SOURCE_URI="${YAM_FOLD_TOWEL_S3_URI:-s3://policy-training/cosmos-policy/fold-blue-towel-twice/dataset/}"
DEST_DIR="${1:-${YAM_FOLD_TOWEL_DATA_DIR:-$HOME/data/fold_blue_towel_twice}}"

command -v aws >/dev/null
mkdir -p "${DEST_DIR}"
aws s3 sync "${SOURCE_URI}" "${DEST_DIR}" --region us-west-2

test -f "${DEST_DIR}/conversion_manifest.json"
test -f "${DEST_DIR}/dataset_statistics.json"
test -f "${DEST_DIR}/dataset_statistics_post_norm.json"
test -f "${DEST_DIR}/t5_embeddings.pkl"

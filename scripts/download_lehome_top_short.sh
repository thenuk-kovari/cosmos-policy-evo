#!/usr/bin/env bash
set -euo pipefail

DEST=${1:?usage: $0 /absolute/path/to/raw_lehome}
mkdir -p "$DEST"
hf download lehome/dataset_challenge --repo-type dataset \
  --include 'record_top_short_release_10/**' \
  --local-dir "$DEST"

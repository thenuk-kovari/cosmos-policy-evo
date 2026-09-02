#!/usr/bin/env bash
set -euo pipefail

: "${RAW_DIR:?Set RAW_DIR to persistent raw dataset storage}"
: "${GENROBOT_EE_Q0_DATA_DIR:?Set GENROBOT_EE_Q0_DATA_DIR to converted output storage}"

UV="${UV:-$HOME/.local/bin/uv}"
WORKERS="${CONVERT_WORKERS:-8}"
TASK="${TASK:-fold the towel}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="$RAW_DIR/Organize_Clutter/fold_towel/00001"

cd "$REPO"
"$UV" sync --extra cu128 --group aloha --python 3.10
"$UV" run --no-sync hf download genrobot2025/10Kh-RealOmin-OpenData --repo-type dataset --include 'Organize_Clutter/fold_towel/00001/*.mcap' --local-dir "$RAW_DIR"
test "$(find "$SOURCE" -maxdepth 1 -name '*.mcap' | wc -l)" -eq 702
mkdir -p "$GENROBOT_EE_Q0_DATA_DIR"
export REPO UV GENROBOT_EE_Q0_DATA_DIR TASK
find "$SOURCE" -maxdepth 1 -name '*.mcap' -print0 | sort -z | xargs -0 -n1 -P "$WORKERS" bash -c '
  "$UV" run --no-sync python "$REPO/projects/genrobot_towel_ee6d/convert_mcap.py" "$1" --output "$GENROBOT_EE_Q0_DATA_DIR" --task "$TASK" || test $? -eq 2
' _
"$UV" run --no-sync python projects/genrobot_towel_ee6d/validate_dataset.py "$GENROBOT_EE_Q0_DATA_DIR" --minimum-episodes 600
"$UV" run --no-sync python projects/evo_q0_towel/prepare_text_embedding.py "$GENROBOT_EE_Q0_DATA_DIR" --task "$TASK"

#!/usr/bin/env bash
set -euo pipefail

: "${LEHOME_ZARR:?set LEHOME_ZARR to the prepared .zarr directory}"
: "${LEHOME_SPLIT_MANIFEST:?set LEHOME_SPLIT_MANIFEST to the generated manifest JSON}"

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python train.py --config-name=train_diffusion_unet_lehome_q0 \
  training.device=cuda:0 \
  hydra.run.dir="data/outputs/$(date +%Y.%m.%d)/$(date +%H.%M.%S)_lehome_top_short_q0"

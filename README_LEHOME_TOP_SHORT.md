# LeHome top-short q0 Diffusion Policy baseline

This is a single-task baseline for `record_top_short_release_10`.

## Contract

- 250 released demonstrations at 30 Hz, split deterministically into 230 training and 20 validation episodes using seed 42.
- Observations are top, left-wrist, and right-wrist RGB plus measured 12D joint/gripper state.
- Labels are future measured state, never the source `action` column.
- The 50-row target from query frame `i` is `state[i + 1 : i + 51] - state[i]`.
- At deployment, reconstruct every joint/gripper target as `live_q0 + predicted_delta`.

## Model

The original bimanual Diffusion Policy paper states that its bimanual tasks used the real Push-T hyperparameters without tuning; it does not release a shirt-specific YAML. This baseline reuses that architecture: independent unpretrained ResNet-18 GroupNorm/spatial-softmax encoders, FiLM-conditioned 1D CNN U-Net widths `[512,1024,2048]`, cosine DDIM with 100 steps, EMA, AdamW at `1e-4`, two observation frames, and 8 executed rows per re-query. The intentional changes are three available LeHome RGB cameras, 12D q0 deltas, and a 50-row 30-Hz prediction horizon. It has 371,600,132 trainable parameters.

## RTX 5090 VM

```bash
git clone --branch training/lehome-top-short-dp https://github.com/thenuk-kovari/video-policy-handoff.git diffusion-policy-lehome
cd diffusion-policy-lehome
./scripts/bootstrap_5090.sh
source .venv/bin/activate
RAW=$HOME/data/lehome_raw
./scripts/download_lehome_top_short.sh "$RAW"
ZARR=$HOME/data/lehome_top_short_q0.zarr
SPLIT=$HOME/data/lehome_top_short_q0_split.json
python scripts/prepare_lehome_top_short.py --raw-root "$RAW" --out "$ZARR" --manifest-out "$SPLIT"
export LEHOME_ZARR="$ZARR"
export LEHOME_SPLIT_MANIFEST="$SPLIT"
./scripts/train_lehome_top_short_5090.sh
```

The conversion decodes AV1 video once into a compressed local RGB Zarr store. Training does not access Hugging Face or decode video. The raw data, Zarr, generated split, checkpoints, and W&B data are ignored by Git.

`training.num_epochs=600` matches the official image-policy config. Batch size is 32 rather than 64 because this has a three-camera, 50-row workload on a single 32-GB GPU. Run a short throughput/memory check first; if there is headroom, override both data-loader batch sizes to 64.

This branch has offline training/validation and checkpointing. Simulator rollouts are intentionally not faked; a LeHome simulator runner is the next separate addition.

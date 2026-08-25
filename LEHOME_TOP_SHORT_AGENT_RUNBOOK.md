# Agent runbook: LeHome top-short q0 Diffusion Policy

This document is the complete handoff for a new agent/operator. The goal is to
train one offline vision Diffusion Policy baseline, not to modify the robot
stack or run deployment. Follow the contract below exactly.

## Scope

Train only the public Hugging Face task:

```text
lehome/dataset_challenge / record_top_short_release_10
```

It has ten source sessions, `001` through `010`, with 25 demonstrations each:
250 total episodes. Do **not** include `pant`, `long`, or any other task folder.
Do not add depth, language, EE/FK losses, or source `action` labels to this
baseline.

## How the raw LeHome data is arranged

LeHome uses LeRobot v3 storage. It is not a ROS bag.

```text
record_top_short_release_10/
  001/ ... 010/                  # each has 25 episodes
  */data/**/*.parquet            # state/action rows at 30 Hz
  */meta/episodes/**/*.parquet   # authoritative episode boundaries
  */videos/.../*.mp4             # shared AV1 video chunks
```

Do not infer an episode boundary from a video-file boundary. One MP4 contains
multiple episodes; `meta/episodes` is the source of truth and the converter
uses it.

Each 30 Hz source frame contains:

| Field | Shape | Use |
|---|---:|---|
| `observation.images.top_rgb` | 480×640×3 RGB | yes, overhead view |
| `observation.images.left_rgb` | 480×640×3 RGB | yes, left wrist view |
| `observation.images.right_rgb` | 480×640×3 RGB | yes, right wrist view |
| `observation.state` | float32[12] | yes, recorded/measured state |
| `action` | float32[12] | **never use** in this run |
| `observation.top_depth` | uint16[480,640] | **never use** in this run |

State order:

```text
0 left_shoulder_pan     6 right_shoulder_pan
1 left_shoulder_lift    7 right_shoulder_lift
2 left_elbow_flex       8 right_elbow_flex
3 left_wrist_flex       9 right_wrist_flex
4 left_wrist_roll      10 right_wrist_roll
5 left_gripper         11 right_gripper
```

## Non-negotiable q0 label contract

This is a q0-anchored **observed-state** policy. “q0” means the recorded
query-time proprioception, not the dataset's action command.

At eligible anchor frame `i`:

```text
input images: top/left/right RGB at i-1 and i
input state:  state[i-1], state[i]; q0 = state[i]

label[k] = state[i + 1 + k] - q0,  k = 0...49
```

Thus each target is `[50,12]` future measured-state deltas, covering 1.67 s
at native 30 Hz. At deployment the matching reconstruction is:

```text
physical_target[k] = live_measured_q0 + predicted_delta[k]
```

Apply that reconstruction to all twelve dimensions, including both grippers.
Do not make grippers absolute model outputs for this experiment.

All q0-state and delta bounds are fit from training episodes only using the
official Diffusion Policy min/max `[-1,+1]` normalizer. Validation data must
never contribute to those bounds.

## Frozen split

The preparation script sorts all 250 IDs as `001/000` through `010/024`, then
uses a NumPy seeded permutation with seed 42 to choose 20 validation episodes.
The remaining 230 are training. It writes the exact IDs to a JSON manifest.

Keep this manifest next to the prepared Zarr store. A different manifest is a
different experiment; do not regenerate one when resuming.

## Model configuration

The upstream paper has no released shirt-specific YAML. It says the bimanual
experiments use the real Push-T hyperparameters without tuning. The local config
`diffusion_policy/config/train_diffusion_unet_lehome_q0.yaml` adapts that model:

| Component | Setting |
|---|---|
| RGB encoders | 3 independent unpretrained ResNet-18s, GroupNorm + spatial softmax |
| Image processing | 480×640 → 240×320 RGB, random 216×288 crop in training |
| Denoiser | FiLM-conditioned temporal CNN U-Net `[512,1024,2048]` |
| Observation history | 2 frames |
| Prediction horizon | 50 rows |
| Execute/requery convention | execute first 8 rows then acquire fresh q0 |
| Scheduler | 100-step squared-cosine DDIM |
| Optimizer | AdamW, lr `1e-4`, cosine schedule, 500 warmup steps |
| Stability | EMA |
| Epochs | 600 |

The model has 371,600,132 trainable parameters. Batch size is intentionally 32
for an RTX 5090's 32 GB because the original 64-batch source config did not
have this exact three-camera, 50-row workload. Test memory before raising it.

## Training from a fresh RTX 5090 VM

Requirements: recent NVIDIA driver, Python 3.10+, and at least 60 GB free local
SSD. Keep raw data, Zarr, checkpoints, W&B cache, and credentials outside Git.

```bash
git clone --branch training/lehome-top-short-dp \
  https://github.com/thenuk-kovari/video-policy-handoff.git diffusion-policy-lehome
cd diffusion-policy-lehome
./scripts/bootstrap_5090.sh
source .venv/bin/activate

RAW=$HOME/data/lehome_raw
./scripts/download_lehome_top_short.sh "$RAW"

ZARR=$HOME/data/lehome_top_short_q0.zarr
SPLIT=$HOME/data/lehome_top_short_q0_split.json
python scripts/prepare_lehome_top_short.py \
  --raw-root "$RAW" --out "$ZARR" --manifest-out "$SPLIT"

export LEHOME_ZARR="$ZARR"
export LEHOME_SPLIT_MANIFEST="$SPLIT"
wandb login  # interactive only; never save a token in the repository
```

Run this mandatory short check before a multi-day job:

```bash
python train.py --config-name=train_diffusion_unet_lehome_q0 \
  training.device=cuda:0 training.max_train_steps=20 training.max_val_steps=4 \
  training.num_epochs=1 hydra.run.dir=data/outputs/smoke
```

Confirm: CUDA is used, no OOM occurs, action tensor is `[batch,50,12]`, a
finite `train_loss` and `val_loss` are logged, and the Zarr preparation reported
exactly `230 train, 20 val`.

Then launch the resumable full run:

```bash
./scripts/train_lehome_top_short_5090.sh
```

The converter decodes AV1 only once into compressed local Zarr. Training should
not contact Hugging Face or decode video per batch. If the memory smoke check
has substantial headroom, a separate controlled retry may override both loader
batch sizes to 64; otherwise keep 32.

## Outputs and checkpoint selection

Hydra creates `data/outputs/...` containing:

```text
logs.json.txt                offline train/validation metrics
checkpoints/latest.ckpt      resumable state
checkpoints/epoch=...ckpt    top 5 validation-loss checkpoints
.hydra/config.yaml           resolved immutable run configuration
```

Pick the lowest `val_loss` checkpoint as the offline candidate. This branch has
no simulator evaluator by design: `NoopImageRunner` produces no fake success
score. A separate simulator adapter must use exactly the RGB transforms,
normalizer, q0 reconstruction, and 8-row receding-horizon execution above.

Resume only by rerunning the same command into the same Hydra run directory;
`training.resume=true` restores `checkpoints/latest.ckpt`.

## Stop and investigate if

- converter says decoded video frame count differs from Parquet state frames;
- the generated split is not exactly 230/20;
- the model receives the source `action` column;
- validation episodes were used for normalization;
- CUDA OOM occurs (lower batch to 16; do not change the q0 contract);
- W&B is unavailable (set `logging.mode=offline`; local logs remain valid).

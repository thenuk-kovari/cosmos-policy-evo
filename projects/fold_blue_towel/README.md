# I2RT YAM blue-towel training

This project fine-tunes Cosmos Predict2 2B to **fold the blue towel twice** from
23 successful bimanual demonstrations. Two additional successful episodes are
held out.

## Durable artifacts

- Source and configuration: this repository
- Dataset: `s3://policy-training/cosmos-policy/fold-blue-towel-twice/dataset/`
- Checkpoints: `s3://policy-training/cosmos-policy-fold-towel/yam/predict2-2b-23demos-75policy-25world/checkpoints/`
- Metrics: W&B project `cosmos-policy-fold-towel`

No credentials, robot videos, HDF5 files, or checkpoints belong in Git.

## Training contract

- Predict2 2B base checkpoint
- 8 GPUs, global batch 200 (25 per GPU)
- 10,000 optimizer steps
- 50-step action chunks
- 75% policy samples: joint `p(a, s' | s)`
- 25% world samples: `p(s' | s, a)`
- No value-prediction loss
- LR `1e-4`; 500-step warmup; linear decay through step 7,500; sharp
  drop to `6e-6`; hold through step 10,000
- Checkpoint every 1,000 steps directly to S3

The value latent remains in the sequence only to retain compatibility with the
stock ALOHA inference path. Its loss is masked.

## Dataset

Run on the host:

```bash
projects/fold_blue_towel/sync_dataset.sh
```

The default local destination is `$HOME/data/fold_blue_towel_twice`. The
conversion manifest records the camera correction, excluded trajectories,
episode split, frequency, and tensor semantics.

## Container setup

Build the pinned official environment:

```bash
sudo docker build -t cosmos-policy docker
```

Mount the repository, dataset, uv cache, uv-managed Python directory, a
persistent output directory, and the two runtime secret files. Inside the
container set:

```bash
export YAM_FOLD_TOWEL_DATA_DIR=/data/fold_blue_towel_twice
export IMAGINAIRE_OUTPUT_ROOT=/outputs
export WANDB_API_KEY
export YAM_S3_CREDENTIALS=/run/secrets/cosmos_s3.json
projects/fold_blue_towel/train.sh
```

`YAM_S3_CREDENTIALS` must point to the Cosmos MSC backend's JSON credential
file. It contains `aws_access_key_id`, `aws_secret_access_key`, `region_name`,
and `endpoint_url`. Mount it read-only into the container. The ordinary AWS SDK
credential chain alone is not sufficient for this backend. Do not put the W&B
key or AWS credentials in this repository, image, or shell script.

## Resume

After at least one S3 checkpoint exists, relaunch with:

```bash
RESUME_FROM_S3=1 projects/fold_blue_towel/train.sh
```

The fixed job project/group/name intentionally gives initial and resumed jobs
the same S3 checkpoint prefix and W&B run identity.

## Preflight

Before the 10K run:

1. Validate the local dataset and T5 embedding.
2. Resolve and inspect the named Hydra experiment.
3. Load the base checkpoint.
4. Run a one-GPU forward/backward update.
5. Run a short eight-GPU distributed smoke test.
6. Save a smoke-test checkpoint to a separate S3 prefix and load it back.

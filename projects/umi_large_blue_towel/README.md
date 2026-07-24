# UMI large blue towel stage

Cosmos Predict2 2B stage-1 training for the instruction:

`fold the large blue towel twice`

## Shared action/proprio contract

Every sample uses 29 dimensions so its checkpoint can be continued with
on-embodiment teleoperation data later.

| Slice | Meaning | UMI action loss |
| --- | --- | --- |
| `0:8` | left 7 joints + gripper | masked |
| `8:16` | right 7 joints + gripper | masked |
| `16` | elevator | masked |
| `17:23` | left EE absolute xyz + rotation vector | supervised |
| `23:29` | right EE absolute xyz + rotation vector | supervised |

The converter reads source `xyzw` quaternions, normalizes/sign-aligns them,
and unwraps equivalent rotation-vector branches across time. Source rows are
resampled from 30 Hz to 25 Hz by nearest timestamp. The base, left-gripper,
and right-gripper cameras are mapped to the three ALOHA camera slots.

## Convert and validate

```bash
source ~/cosmos-policy-venv/bin/activate
cd ~/cosmos-policy
python projects/umi_large_blue_towel/convert_lerobot_parquet.py \
  --source ~/data/umi_yubi50_source \
  --reviews projects/umi_large_blue_towel/reviews.json \
  --output ~/data/umi_large_blue_towel \
  --overwrite

python projects/umi_large_blue_towel/validate_dataset.py \
  ~/data/umi_large_blue_towel
```

The review manifest must contain all 50 episodes, with 48 marked `keep`.

## Train

The production schedule is 8,000 steps, per-rank batch 25 (effective global batch 200 on eight GPUs),
75% action prediction / 25% future-state prediction, 400-step warmup, decay
through step 6,000, and checkpoints every 1,000 steps.

```bash
cd ~/cosmos-policy
export UMI_LARGE_BLUE_TOWEL_DATA_DIR=~/data/umi_large_blue_towel
./projects/umi_large_blue_towel/train.sh
```

The launcher logs online to W&B and saves checkpoints to the configured
`policy-training` object store. `MAX_ITER`, `SAVE_ITER`, `RUN_NAME`,
`WANDB_MODE`, and `SAVE_TO_S3` can be overridden for smoke tests or resumes.

# EVO observed-state q0 training

This project intentionally does not use recorded command topics. At a sampled
camera/state frame `i`, the loader builds a 50-row action chunk from future
measured states:

```text
q0 = measured_state[i]
future[k] = measured_state[min(i + k + 1, T - 1)]

arm_delta[k] = future_arm[k] - q0_arm
elevator_delta[k] = future_elevator[k] - q0_elevator
gripper_target[k] = future_gripper[k]
```

The layout is left-first and 17-D:

```text
0:7    left joint query-relative deltas, radians
7      left radial-gripper absolute measured joint position, radians
8:15   right joint query-relative deltas, radians
15     right radial-gripper absolute measured joint position, radians
16     elevator query-relative delta, metres
```

Rows are sampled at 30 Hz. Row zero is the next frame, not the observation
frame, so the 50-row horizon is 1.667 seconds. Every row uses the same q0.

## Convert

Install the Cosmos ALOHA dependencies, then run:

```bash
python projects/evo_q0_towel/prepare_mcap.py \
  --raw-root /path/to/mcaps \
  --out /persistent/evo_q0_towel
```

Use `--mapping selected.json` for an explicit ordered JSON list of MCAP paths,
or `--limit 20` for a deterministic smoke/pilot subset of the sorted files.
Conversion fails on missing state dimensions, large state/camera gaps,
unsupported images, non-finite values, or episodes shorter than 51 frames.

Validate before allocating GPUs:

```bash
python projects/evo_q0_towel/validate_dataset.py /persistent/evo_q0_towel
```

For staged training, compute one normalization contract over every episode that
may appear in either stage. Do this before stage one so loading stage two does
not change what a normalized model output means:

```bash
python projects/evo_q0_towel/compute_statistics.py \
  --data-dir /persistent/evo_q0_pilot20 \
  --data-dir /persistent/evo_q0_new50 \
  --out /persistent/evo_q0_canonical_stats
```

Set `EVO_Q0_STATS_DIR=/persistent/evo_q0_canonical_stats` in every stage. The
resulting `dataset_statistics.json` must travel with the final checkpoint for
inference. Each dataset root also needs `t5_embeddings.pkl` containing the key
`fold the blue towel twice`; it can be copied after it is computed once.

```bash
python projects/evo_q0_towel/prepare_text_embedding.py /persistent/evo_q0_pilot20
cp /persistent/evo_q0_pilot20/t5_embeddings.pkl /persistent/evo_q0_new50/
```

## Train on 8 GPUs

The default experiment is 75% action-only and 25% future-state-only. It never
creates value-prediction samples.

```bash
export EVO_Q0_DATA_DIR=/persistent/evo_q0_towel
export EVO_Q0_STATS_DIR=/persistent/evo_q0_canonical_stats
export IMAGINAIRE_OUTPUT_ROOT=/persistent/cosmos-output
export JOB_NAME=evo-q0-pilot20
export MAX_STEPS=2000
./projects/evo_q0_towel/train.sh
```

To initialize a later run from a checkpoint while resetting optimizer and LR
state:

```bash
export EVO_Q0_DATA_DIR=/persistent/evo_q0_new50
export LOAD_PATH='object-store/path/to/checkpoints/iter_000002000'
export LOAD_TRAINING_STATE=false
export JOB_NAME=evo-q0-stage2
export MAX_STEPS=6000
export EVO_Q0_STATS_DIR=/persistent/evo_q0_canonical_stats
./projects/evo_q0_towel/train.sh
```

Set `LOAD_TRAINING_STATE=true` only for a literal interruption/resumption of
the same optimization schedule and dataset. Dataset adaptation should normally
load weights but reset optimizer/LR state.

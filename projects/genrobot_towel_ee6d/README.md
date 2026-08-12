# GenRobot towel transfer: q0-local EE-6D stage

This stage trains on the gated Hugging Face subtree
`Organize_Clutter/fold_towel/00001` (702 MCAP episodes, about 36.4 GB raw).
It intentionally contains no Evo replay. Evo teleop is a later stage.

## Contract

The shared action head is 35-D and left-first:

| Dims | Meaning | UMI loss |
|---|---|---|
| 0:3 | left q0-body translation delta, metres | yes |
| 3:9 | left relative rotation, first two matrix columns | yes |
| 9:12 | right q0-body translation delta, metres | yes |
| 12:18 | right relative rotation, first two matrix columns | yes |
| 18:25 | left joint q0 delta, radians | masked |
| 25 | left canonical gripper aperture, 0 closed / 1 open | yes |
| 26:33 | right joint q0 delta, radians | masked |
| 33 | right canonical gripper aperture, 0 closed / 1 open | yes |
| 34 | elevator q0 delta, metres | masked |

For each hand and one frozen query pose `(p0, R0)`, row `k` is:

```text
dp[k] = R0^T (p[k] - p0)
dR[k] = R0^T R[k]
rot6d[k] = [dR[:,0], dR[:,1]]
```

Rows are future samples `q0+1 ... q0+50`, terminal padded. Translation
and rotation therefore do not depend on either DAS VIO's arbitrary world
origin. They are not cumulative step-to-step deltas.

The MCAP pose is the DAS `base_link`. The released URDF has no defined TCP, so
conversion defaults to identity `base_link -> action_frame` and records that
choice in every HDF5. Do not silently call it a TCP. If calibration is obtained,
pass the fixed 7-value transforms (`tx ty tz qx qy qz qw`) to the converter and
reconvert. Stage-two Evo FK must use the analogous physical frame.

The shared 17-D proprio layout is left joints7, left aperture, right joints7,
right aperture, elevator. UMI provides only the two apertures; the other 15
slots are zero placeholders and are excluded from future-proprio loss by
`proprio_dim_mask`.

Normalization uses fixed physical bounds in
`bimanual_shared35_fixed_physical_v1`, not empirical extrema. This keeps the
numerical meaning unchanged when Evo supervision is added later. The converter
hard-checks sampled q0 chunks against those bounds.

## Source mapping

- robot0 VIO/base camera/gripper width -> left hand
- robot1 VIO/base camera/gripper width -> right hand
- `/vio/eef_pose`: interpolated translation + quaternion SLERP
- `/sensor/magnetic_encoder`: width in metres, mapped by `width / 0.103`
- `/sensor/camera0/compressed`: H.264 wrist video
- absent third-person camera: explicit black video (never duplicate a wrist
  view under a false camera identity). `GenRobotEEQ0Dataset` marks its current
  and future latent indices unavailable, so this placeholder contributes
  neither conditioning nor future-image loss. Only the two real wrist-camera
  views are supervised; the placeholder merely preserves the shared
  three-camera sequence shape for the later Evo teleop stage.
- prompt: `fold the towel`

The training image path matches existing Cosmos preprocessing: RGB video is
loaded and directly resized to 224x224; training augmentation then applies a
shared 90% random resized crop, color jitter, and small rotation. There is no
center crop or letterboxing.

## Prepare on the 8xH100 VM

The Hugging Face account/token must already have access to the dataset.
Use persistent disks for both directories. Conversion is CPU-heavy and the
converted videos are larger than the 36.4 GB download.

```bash
export RAW_DIR=/persistent/genrobot-raw
export GENROBOT_EE_Q0_DATA_DIR=/persistent/genrobot-towel-ee6d35
export CONVERT_WORKERS=8
./projects/genrobot_towel_ee6d/prepare.sh
```

The numeric split is deterministic: every 10th episode is validation; the
remaining episodes are training. Each episode is independently audited and
rejections are recorded in `conversion_manifest.json`. Preparation requires at
least 600 accepted episodes and writes the fixed statistics plus the single T5
embedding.

## Train

```bash
export GENROBOT_EE_Q0_DATA_DIR=/persistent/genrobot-towel-ee6d35
export IMAGINAIRE_OUTPUT_ROOT=/persistent/cosmos-output
export JOB_NAME=genrobot-towel-ee6d35-12k
./projects/genrobot_towel_ee6d/train.sh
```

Default: 8 GPUs, 12,000 iterations, checkpoint every 500 steps. The data mixture
matches the prior q0/YAM recipe: a fixed 3:1 population is shuffled by the
distributed sampler, giving 75% demo samples with action + future-state loss
and 25% copied-success samples with future-state-only loss. It is interleaved,
not a first-quarter/last-three-quarters curriculum.

# Jetson Thor Cosmos Policy Triton image

This image embeds the exact 17D/17D q0 Cosmos Policy source and replaces Evo
Triton slot B with a strict Cosmos adapter. It does not contain checkpoints,
datasets, credentials, normalization statistics, T5 embeddings, or tokenizer weights.

Build:

```bash
docker build \
  -f docker/thor/Dockerfile \
  -t cosmos-policy-evo-q0-triton:recovery .
```

The runtime checkpoint directory must contain:

```text
model/.metadata
model/__*.distcp
dataset_statistics.json
t5_embeddings.pkl
tokenizer.pth
```

Write its container-visible directory into the standard Evo marker:

```text
/home/developer/evo_ws/src/evo/policies/.slot_b_path
```

For one-URI startup with the Evo entrypoint, upload that directory as one S3
prefix and set:

```bash
export EVO_CKPT_S3_URI__SLOT_B=s3://your-private-bucket/evo-q0-runtime/
```

The entrypoint downloads it to `/models/slot_b/checkpoint`. A standard marker
file still overrides that default when policies are staged manually.

The native output is one full action chunk with shape `[50,17]`. The contract
is left-first and arm and elevator channels are fixed-q0 deltas; gripper channels are absolute. See
`deploy/triton/policy_config.json` for every dimension.

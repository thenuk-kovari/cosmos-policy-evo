ARG TRITON_IMAGE=815254799754.dkr.ecr.us-west-2.amazonaws.com/evo/dev/triton-inference-server:latest@sha256:be28f0cd51b0ca3d363ddfdde08a07dcfb366476a9103678ebe4146d22bb558f
FROM ${TRITON_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=5 \
    NVTE_FRAMEWORK=pytorch \
    COSMOS_POLICY_SKIP_CONFIG_CHECKPOINT_DOWNLOAD=1 \
    PYTHONPATH=/opt/cosmos-policy:/opt/triton:${PYTHONPATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/thor/requirements-runtime.txt /tmp/cosmos-runtime-requirements.txt
RUN pip install --no-cache-dir --break-system-packages \
        -r /tmp/cosmos-runtime-requirements.txt \
    && pip install --no-cache-dir --break-system-packages --no-deps \
        megatron-core==0.14.0 \
    && rm -rf /root/.cache/pip /tmp/cosmos-runtime-requirements.txt

# Transformer Engine 2.13+ requires a cuBLASLt symbol introduced after the
# CUDA 13.0 libraries deployed on some Thor Evo images. Version 2.12 retains
# the required RMSNorm, attention, and rotary APIs without that newer ABI.
RUN pip install --no-cache-dir --break-system-packages \
        transformer-engine==2.12.0 transformer-engine-cu13==2.12.0 \
    && rm -rf /root/.cache/pip

RUN pip install --no-cache-dir --break-system-packages \
        fastapi==0.121.2 \
        nvdlfw-inspect==0.2.2 \
        onnx==1.20.0 onnx-ir==0.1.12 onnxscript==0.5.6 \
        triton==3.7.1 \
        pytz==2025.2 \
        uvicorn==0.51.0 \
    && rm -rf /root/.cache/pip

# Select torchvision only after all dependency-resolving installs, using the
# actually imported vendor Torch ABI rather than potentially stale metadata.
RUN TORCH_VERSION="$(/usr/bin/python3 -c 'import torch; print(torch.__version__.split("+", 1)[0])')" \
    && case "$TORCH_VERSION" in \
         2.11.0) TORCHVISION_VERSION=0.26.0 ;; \
         2.13.0) TORCHVISION_VERSION=0.28.0 ;; \
         *) echo "Unsupported final Torch version: $TORCH_VERSION" >&2; exit 1 ;; \
       esac \
    && pip install --no-cache-dir --break-system-packages --no-deps --force-reinstall \
         "torchvision==${TORCHVISION_VERSION}" \
    && /usr/bin/python3 -c 'import torch, torchvision; assert hasattr(torch.ops.torchvision, "nms")' \
    && rm -rf /root/.cache/pip

# Build the framework extension only after the final Torch ABI is established;
# later compilation would otherwise leave unresolved c10 symbols at runtime.
RUN pip install --no-cache-dir --break-system-packages --no-build-isolation --no-deps \
        transformer-engine-torch==2.12.0 \
    && rm -rf /root/.cache/pip

COPY . /opt/cosmos-policy
RUN pip install --no-cache-dir --break-system-packages --no-deps /opt/cosmos-policy \
    && find /opt/cosmos-policy -type d -name __pycache__ -prune -exec rm -rf '{}' + \
    && rm -rf /root/.cache/pip \
    && /usr/bin/python3 -c 'import torch, torchvision; assert hasattr(torch.ops.torchvision, "nms")' \
    && /usr/bin/python3 -c 'import torch; torch.compile=lambda fn=None,*a,**k: fn if fn is not None else (lambda f:f); from cosmos_policy.experiments.robot.cosmos_utils import get_action, get_model'

COPY deploy/triton/backends/cosmos_policy_backend.py /opt/triton/backends/cosmos_policy_backend.py
COPY deploy/triton/model_repository/slot_b/ /models/slot_b/

ARG SOURCE_REVISION=uncommitted

LABEL org.opencontainers.image.source="https://github.com/thenuk-kovari/video-policy-handoff" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      ai.kovari.policy.training-baseline="a38079969d4dd29a5dedda746764c84d9a1c7892" \
      ai.kovari.policy.contract="evo_q0_observed_state_left_first_v1" \
      ai.kovari.policy.action-dim="17" \
      ai.kovari.policy.proprio-dim="17" \
      ai.kovari.policy.chunk-size="50"

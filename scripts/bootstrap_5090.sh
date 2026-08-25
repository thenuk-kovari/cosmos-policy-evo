#!/usr/bin/env bash
set -euo pipefail

# RTX 5090 / Blackwell needs a current PyTorch CUDA wheel.  Use a clean venv;
# do not create the paper's 2023 CUDA-11.6 conda environment on this machine.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
python -m pip install -r requirements-5090.txt
python -m pip install -e .
python - <<'PY'
import torch, torchvision
print("torch", torch.__version__, "torchvision", torchvision.__version__)
print("cuda available", torch.cuda.is_available(), "device", torch.cuda.get_device_name(0))
PY

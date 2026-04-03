#!/bin/bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_VERSION="${TORCH_VERSION:-2.1.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.16.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.1.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
INSTALL_APT_DEPS="${INSTALL_APT_DEPS:-1}"
DOWNLOAD_WEIGHTS="${DOWNLOAD_WEIGHTS:-1}"
RUN_VERIFY="${RUN_VERIFY:-1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${VENV_DIR}" = /* ]]; then
  VENV_PATH="${VENV_DIR}"
else
  VENV_PATH="${ROOT_DIR}/${VENV_DIR}"
fi

log() {
  echo
  echo "==> $1"
}

die() {
  echo "ERROR: $1" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<EOF
Usage: ./install_ubuntu_h100_venv.sh

Optional environment variables:
  PYTHON_BIN=python3.10
  VENV_DIR=.venv
  TORCH_VERSION=2.1.0
  TORCHVISION_VERSION=0.16.0
  TORCHAUDIO_VERSION=2.1.0
  TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
  INSTALL_APT_DEPS=1
  DOWNLOAD_WEIGHTS=1
  RUN_VERIFY=1
  HF_ENDPOINT=<optional mirror>

Examples:
  ./install_ubuntu_h100_venv.sh
  VENV_DIR=.venv-musetalk ./install_ubuntu_h100_venv.sh
  PYTHON_BIN=python3 INSTALL_APT_DEPS=0 ./install_ubuntu_h100_venv.sh
EOF
  exit 0
fi

log "Checking base requirements"
require_command bash
require_command git
require_command "${PYTHON_BIN}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "WARNING: nvidia-smi not found. GPU verification will be limited."
fi

if [[ -f /etc/os-release ]]; then
  . /etc/os-release
  echo "Detected OS: ${PRETTY_NAME:-unknown}"
fi

if [[ "${INSTALL_APT_DEPS}" == "1" ]]; then
  log "Installing Ubuntu system packages"
  require_command sudo
  sudo apt-get update
  sudo apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    cmake \
    ninja-build
fi

log "Creating virtual environment at ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_PATH}"

log "Activating virtual environment"
source "${VENV_PATH}/bin/activate"

log "Upgrading packaging tools"
python -m pip install -U pip wheel
python -m pip install "setuptools<81"

log "Installing PyTorch"
pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "${TORCH_INDEX_URL}"

log "Installing MuseTalk Python requirements"
pip install -r "${ROOT_DIR}/requirements.txt"

log "Installing OpenMMLab dependencies"
pip install --no-cache-dir -U openmim
pip install "mmengine==0.10.7"
pip install "mmcv==2.1.0" -f "https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html"
pip install "mmdet==3.2.0"
pip install "mmpose==1.2.0"

log "Reapplying critical repo pins"
pip install --upgrade --force-reinstall \
  "numpy==1.23.5" \
  "opencv-python==4.9.0.80" \
  "huggingface_hub==0.30.2" \
  "transformers==4.39.2"

if [[ "${DOWNLOAD_WEIGHTS}" == "1" ]]; then
  log "Downloading model weights"
  bash "${ROOT_DIR}/download_weights.sh"
fi

if [[ "${RUN_VERIFY}" == "1" ]]; then
  log "Running verification"
  python - <<'PY'
import os
import torch
import mmcv
import mmdet
import mmpose

print("python:", os.sys.executable)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu count:", torch.cuda.device_count())
    print("gpu0:", torch.cuda.get_device_name(0))
print("mmcv:", mmcv.__version__)
print("mmdet:", mmdet.__version__)
print("mmpose:", mmpose.__version__)

required_paths = [
    "models/musetalkV15/unet.pth",
    "models/musetalkV15/musetalk.json",
    "models/sd-vae/config.json",
    "models/whisper/config.json",
    "models/dwpose/dw-ll_ucoco_384.pth",
    "models/syncnet/latentsync_syncnet.pt",
    "models/face-parse-bisent/79999_iter.pth",
    "models/face-parse-bisent/resnet18-5c106cde.pth",
]
missing = [path for path in required_paths if not os.path.exists(path)]
if missing:
    print("Missing model files:")
    for path in missing:
        print(" -", path)
else:
    print("All required model files are present.")
PY
fi

log "Install complete"
cat <<EOF
Next steps:
  source ${VENV_PATH}/bin/activate
  cd ${ROOT_DIR}
  sh inference.sh v1.5 normal

If you want fp16 inference on a single GPU:
  python -m scripts.inference \\
    --inference_config configs/inference/test.yaml \\
    --result_dir results/test \\
    --unet_model_path models/musetalkV15/unet.pth \\
    --unet_config models/musetalkV15/musetalk.json \\
    --version v15 \\
    --gpu_id 0 \\
    --use_float16
EOF

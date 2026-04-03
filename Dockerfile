FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    cmake \
    ninja-build \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/MuseTalk

COPY requirements.txt download_weights.sh install_ubuntu_h100_venv.sh ./

RUN chmod +x install_ubuntu_h100_venv.sh download_weights.sh && \
    PYTHON_BIN=python3 \
    VENV_DIR=/opt/venv \
    INSTALL_APT_DEPS=0 \
    DOWNLOAD_WEIGHTS=0 \
    RUN_VERIFY=0 \
    ./install_ubuntu_h100_venv.sh

COPY . .

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/workspace/.cache/huggingface

CMD ["/bin/bash"]

# Install Instruction

This document captures the stable installation path we validated for Ubuntu servers using `python -m venv`, CUDA 12.x drivers, and NVIDIA H100 GPUs.

It also documents the dependency conflicts we hit during setup and the fixes already folded into this repository.

It now also includes a Docker workflow and validated multi-GPU training presets for 8x H100 systems.

## Recommended Environment

- OS: Ubuntu 22.04 or newer
- Python: 3.10
- Environment manager: `python -m venv`
- GPU: NVIDIA H100
- PyTorch: `2.1.0` with `cu121` wheels
- OpenMMLab stack:
  - `mmengine==0.10.7`
  - `mmcv==2.1.0`
  - `mmdet==3.2.0`
  - `mmpose==1.2.0`

This combination avoids building `mmcv` from source and works better on modern H100 systems than the older `torch 2.0.1 + mmcv 2.0.1` path in the original README.

## One-Command Install

From the repository root:

```bash
chmod +x install_ubuntu_h100_venv.sh
./install_ubuntu_h100_venv.sh
```

The script will:

- create a virtual environment at `.venv`
- install system packages such as `ffmpeg`
- install PyTorch and the OpenMMLab dependencies
- install project requirements
- download model weights
- reapply fragile version pins such as `numpy` and `huggingface_hub`
- run a quick verification step

## Docker

### Build the image

From the repository root:

```bash
docker build -t musetalk:cu121 .
```

The image installs the validated Ubuntu + `venv` software stack but does not bake model weights into the image.

### Run the container

The simplest interactive run command is:

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  -v $(pwd)/models:/workspace/MuseTalk/models \
  -v $(pwd)/data:/workspace/MuseTalk/data \
  -v $(pwd)/results:/workspace/MuseTalk/results \
  -v $(pwd)/exp_out:/workspace/MuseTalk/exp_out \
  musetalk:cu121
```

This drops you into a shell inside the container at `/workspace/MuseTalk`.

### Run the container in the background

```bash
docker run -d \
  --name musetalk \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  -v $(pwd)/models:/workspace/MuseTalk/models \
  -v $(pwd)/data:/workspace/MuseTalk/data \
  -v $(pwd)/results:/workspace/MuseTalk/results \
  -v $(pwd)/exp_out:/workspace/MuseTalk/exp_out \
  musetalk:cu121 \
  sleep infinity
```

Then open a shell in the running container:

```bash
docker exec -it musetalk bash
```

### First-time setup inside the container

If the `models` directory on the host is still empty, download weights once:

```bash
bash download_weights.sh
```

Because `models` is mounted from the host, the downloaded weights persist across container restarts.

### Run inference inside the container

```bash
sh inference.sh v1.5 normal
```

## Multi-GPU Training

The repository now includes validated 8x H100 presets:

- `configs/training/gpu_8xh100.yaml`
- `configs/training/stage1_8xh100.yaml`
- `configs/training/stage2_8xh100.yaml`

These settings are tuned for a single machine with 8 GPUs and enable:

- `MULTI_GPU` launch through `accelerate`
- `bf16` mixed precision
- pinned-memory dataloaders
- conservative batch sizing for 80 GB H100 systems

### Stage 1 on 8x H100

```bash
ACCELERATE_CONFIG=./configs/training/gpu_8xh100.yaml \
TRAIN_CONFIG=./configs/training/stage1_8xh100.yaml \
sh train.sh stage1
```

### Stage 2 on 8x H100

```bash
ACCELERATE_CONFIG=./configs/training/gpu_8xh100.yaml \
TRAIN_CONFIG=./configs/training/stage2_8xh100.yaml \
sh train.sh stage2
```

### Notes on the 8x H100 presets

- `stage1_8xh100.yaml` uses `train_bs: 16` per GPU with `bf16`
- `stage2_8xh100.yaml` uses `train_bs: 1` and `gradient_accumulation_steps: 8` to stay within 80 GB GPU memory more safely
- `train.sh` now accepts `ACCELERATE_CONFIG`, `TRAIN_CONFIG`, and `MAIN_PROCESS_PORT`
- `train.py` now honors `solver.mixed_precision`

If you want to use fewer GPUs, either:

- edit `configs/training/gpu_8xh100.yaml` and change `gpu_ids` plus `num_processes`
- or create a copy with the GPU subset you want to run

## Manual Install

### 1. Install system packages

```bash
sudo apt-get update
sudo apt-get install -y \
  ffmpeg \
  libgl1 \
  libglib2.0-0 \
  build-essential \
  cmake \
  ninja-build
```

### 2. Create and activate a virtual environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel
python -m pip install "setuptools<81"
```

### 3. Install PyTorch

```bash
pip install \
  torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install project requirements

```bash
pip install -r requirements.txt
```

### 5. Install OpenMMLab packages

```bash
pip install --no-cache-dir -U openmim
pip install "mmengine==0.10.7"
pip install "mmcv==2.1.0" \
  -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
pip install "mmdet==3.2.0"
pip install "mmpose==1.2.0"
```

### 6. Reapply fragile version pins

Some OpenMMLab or model-download steps may upgrade transitive dependencies past versions that this repository expects.

Run this once after installation:

```bash
pip install --upgrade --force-reinstall \
  "numpy==1.23.5" \
  "opencv-python==4.9.0.80" \
  "huggingface_hub==0.30.2" \
  "transformers==4.39.2"
```

### 7. Download model weights

```bash
bash download_weights.sh
```

The download script has been updated so it does not upgrade `huggingface_hub` past `1.0`.

### 8. Verify the environment

```bash
python - <<'PY'
import numpy
import cv2
import torch
import huggingface_hub
import transformers
import mmcv
import mmdet
import mmpose

print("numpy:", numpy.__version__)
print("cv2:", cv2.__version__)
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu count:", torch.cuda.device_count())
    print("gpu0:", torch.cuda.get_device_name(0))
print("huggingface_hub:", huggingface_hub.__version__)
print("transformers:", transformers.__version__)
print("mmcv:", mmcv.__version__)
print("mmdet:", mmdet.__version__)
print("mmpose:", mmpose.__version__)
PY
```

Expected key versions:

- `numpy: 1.23.5`
- `opencv-python: 4.9.0.80`
- `huggingface_hub: 0.30.2`
- `transformers: 4.39.2`
- `torch: 2.1.0`
- `mmcv: 2.1.0`
- `mmdet: 3.2.0`
- `mmpose: 1.2.0`

## Run Inference

After installation:

```bash
source .venv/bin/activate
sh inference.sh v1.5 normal
```

For direct invocation with fp16 on GPU 0:

```bash
python -m scripts.inference \
  --inference_config configs/inference/test.yaml \
  --result_dir results/test \
  --unet_model_path models/musetalkV15/unet.pth \
  --unet_config models/musetalkV15/musetalk.json \
  --version v15 \
  --gpu_id 0 \
  --use_float16
```

## Run Training

Default training launcher:

```bash
sh train.sh stage1
sh train.sh stage2
```

8x H100 training launcher:

```bash
ACCELERATE_CONFIG=./configs/training/gpu_8xh100.yaml \
TRAIN_CONFIG=./configs/training/stage1_8xh100.yaml \
sh train.sh stage1

ACCELERATE_CONFIG=./configs/training/gpu_8xh100.yaml \
TRAIN_CONFIG=./configs/training/stage2_8xh100.yaml \
sh train.sh stage2
```

## Troubleshooting

### `ImportError: huggingface-hub>=0.19.3,<1.0 is required`

Cause:
- `huggingface_hub` was upgraded beyond `1.0`, often by a download step.

Fix:

```bash
pip install --upgrade --force-reinstall \
  "huggingface_hub==0.30.2" \
  "transformers==4.39.2"
```

### `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`

Cause:
- `numpy` was upgraded to 2.x and `opencv-python` was still built against NumPy 1.x.

Fix:

```bash
pip uninstall -y numpy opencv-python opencv-python-headless
pip install --no-cache-dir "numpy==1.23.5" "opencv-python==4.9.0.80"
```

### `FileNotFoundError: ./models/dwpose/dw-ll_ucoco_384.pth can not be found`

Fix:

```bash
mkdir -p models/dwpose
huggingface-cli download yzd-v/DWPose \
  --local-dir models/dwpose \
  --include "dw-ll_ucoco_384.pth"
```

### `OSError: Error no file named config.json found in directory models/sd-vae`

Fix:

```bash
mkdir -p models/sd-vae
huggingface-cli download stabilityai/sd-vae-ft-mse \
  --local-dir models/sd-vae \
  --include "config.json" "diffusion_pytorch_model.bin"
```

### TensorFlow prints CUDA driver warnings during inference

These messages are usually not the blocker for MuseTalk inference. MuseTalk uses PyTorch for the actual model path. If PyTorch sees your GPU correctly, these TensorFlow warnings can typically be ignored.

### Training uses the wrong GPU count or wrong GPU IDs

Check the accelerate config you are using:

- `configs/training/gpu.yaml`
- `configs/training/gpu_8xh100.yaml`

Make sure `gpu_ids` and `num_processes` match the GPUs you want to use.

You can also override the launcher config when calling `train.sh`:

```bash
ACCELERATE_CONFIG=./configs/training/gpu_8xh100.yaml \
TRAIN_CONFIG=./configs/training/stage1_8xh100.yaml \
MAIN_PROCESS_PORT=29503 \
sh train.sh stage1
```

### `mmcv` tries to build from source and fails

Use the validated version set in this document. In particular:

- use `torch==2.1.0`
- use `mmcv==2.1.0`
- install `mmcv` from the OpenMMLab wheel index for `cu121/torch2.1.0`
- use `mmdet==3.2.0` and `mmpose==1.2.0`

## Repository Changes Included

The repository now includes:

- `install_ubuntu_h100_venv.sh` for `python -m venv` based setup
- `install_ubuntu_h100.sh` for Conda-based setup
- `Dockerfile` and `.dockerignore` for containerized setup
- an updated `download_weights.sh` that keeps `huggingface_hub` below `1.0`
- post-install repinning of critical packages in the install scripts
- `gpu_8xh100.yaml`, `stage1_8xh100.yaml`, and `stage2_8xh100.yaml` for multi-GPU training
- a training path that now honors `solver.mixed_precision`

#!/bin/bash

set -euo pipefail

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
  echo "Usage: $0 <image_or_video_path> <wav_path> [output_filename.mp4]"
  echo "Example: $0 /data/face.png /data/speech.wav face_speech.mp4"
  exit 1
fi

INPUT_PATH=$1
AUDIO_PATH=$2
OUTPUT_NAME=${3:-}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_CONFIG_PATH="${ROOT_DIR}/configs/inference/test.custom.yaml"

if [ ! -e "$INPUT_PATH" ]; then
  echo "Input image/video not found: $INPUT_PATH"
  exit 1
fi

if [ ! -f "$AUDIO_PATH" ]; then
  echo "Audio file not found: $AUDIO_PATH"
  exit 1
fi

if [[ "$AUDIO_PATH" != *.wav ]]; then
  echo "Audio file must be a .wav file: $AUDIO_PATH"
  exit 1
fi

python3 - <<PY
from pathlib import Path

input_path = Path(${INPUT_PATH@Q}).resolve()
audio_path = Path(${AUDIO_PATH@Q}).resolve()
output_name = ${OUTPUT_NAME@Q}
config_path = Path(${TMP_CONFIG_PATH@Q})

lines = [
    "task_0:",
    f'  video_path: "{input_path}"',
    f'  audio_path: "{audio_path}"',
]
if output_name:
    lines.append(f'  result_name: "{output_name}"')

config_path.write_text("\\n".join(lines) + "\\n")
PY

python3 -m scripts.inference \
  --inference_config "$TMP_CONFIG_PATH" \
  --result_dir "${ROOT_DIR}/results/test" \
  --unet_model_path "${ROOT_DIR}/models/musetalkV15/unet.pth" \
  --unet_config "${ROOT_DIR}/models/musetalkV15/musetalk.json" \
  --version v15 \
  --gpu_id 0 \
  --use_float16

if [ -n "$OUTPUT_NAME" ]; then
  echo "Output saved to ${ROOT_DIR}/results/test/v15/${OUTPUT_NAME}"
else
  INPUT_BASENAME=$(basename "$INPUT_PATH")
  INPUT_BASENAME="${INPUT_BASENAME%.*}"
  AUDIO_BASENAME=$(basename "$AUDIO_PATH")
  AUDIO_BASENAME="${AUDIO_BASENAME%.*}"
  echo "Output saved to ${ROOT_DIR}/results/test/v15/${INPUT_BASENAME}_${AUDIO_BASENAME}.mp4"
fi

#!/bin/bash
# Run PASC on one image and print the tokens it judged least visually grounded,
# which of them triggered a crop, and what evidence each crop produced.

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export HF_HOME="${PROJECT_ROOT}/.cache"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

IMAGE="${1:-${PROJECT_ROOT}/data/RH-Bench/per_images/0.jpg}"
QUESTION="${2:-How many traffic lights are there in the image?}"

echo "=========================================================="
echo "PASC: PAS-gated, attention-cropped self-correction"
echo "image:    ${IMAGE}"
echo "question: ${QUESTION}"
echo "=========================================================="

python "${PROJECT_ROOT}/examples/pasc_demo.py" \
  --model weights/Qwen2.5-VL-7B-Instruct \
  --image "${IMAGE}" \
  --question "${QUESTION}" \
  --dump-flagged

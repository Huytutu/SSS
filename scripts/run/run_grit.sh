#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Get the root directory of the project (one level up from the scripts/ directory)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Configure python path to find the 'ecrd' module
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Configure Hugging Face to store cached models and datasets in the current directory's .cache folder
export HF_HOME="${PROJECT_ROOT}/.cache"

echo "=========================================================="
echo "Running ECRD + GRIT (Uncertainty-triggered Visual Decider)"
echo "Base Model: Qwen/Qwen2.5-VL-7B-Instruct"
echo "GRIT Model: yfan1997/GRIT-20-Qwen2.5-VL-3B"
echo "Hugging Face Cache: ${HF_HUB_CACHE}"
echo "=========================================================="
echo ""

python "${PROJECT_ROOT}/examples/qwen2_5_vl_ecrd_demo.py" \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --image "${PROJECT_ROOT}/assets/fig1_reasoning_pattern.png" \
  --question "Describe the layout and reasoning patterns shown in this diagram." \
  --use-grit \
  --grit-model yfan1997/GRIT-20-Qwen2.5-VL-3B \
  --load-in-4bit \
  --grit-in-4bit \
  --grit-device cpu

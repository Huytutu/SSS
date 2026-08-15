#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Get the root directory of the project (one level up from the eval/ directory)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# inference_treebench_ecrd.py imports "SSS.ecrd", so PYTHONPATH needs the
# directory that CONTAINS the SSS/ package (i.e. the MORAI/ root).
MORAI_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
export PYTHONPATH="${MORAI_ROOT}:${PYTHONPATH}"

# Configure Hugging Face to store cached models and datasets in the current directory's .cache folder
export HF_HOME="${PROJECT_ROOT}/.cache"

echo "=========================================================="
echo "Starting ECRD evaluation on TreeBench"
echo "=========================================================="
echo ""

# Run evaluation -- pass --device / --grit-device / --delta / --limit
python "${PROJECT_ROOT}/eval/inference_treebench_ecrd.py" \
  "$@"

# CUDA_VISIBLE_DEVICES=3,4 bash eval/inference_treebench_ecrd.sh --device 0 --grit-device 1

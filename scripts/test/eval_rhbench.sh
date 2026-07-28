#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Get the root directory of the project (one level up from the scripts/test/ directory)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Configure python path to find the 'ecrd' module
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Configure Hugging Face to store cached models and datasets in the current directory's .cache folder
export HF_HOME="${PROJECT_ROOT}/.cache"

echo "=========================================================="
echo "Starting ECRD evaluation on RH-Bench"
echo "Hugging Face Cache: ${HF_HOME}"
echo "=========================================================="
echo ""

# Run evaluation (you can add --use-grit to test ECRD + GRIT)
python "${PROJECT_ROOT}/scripts/test/eval_rhbench.py" \
  "$@"


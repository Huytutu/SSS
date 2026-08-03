#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Get the root directory of the project (one level up from the scripts/test/ directory)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# eval_treebench.py imports "MORAI.SSS.ecrd", so PYTHONPATH needs the directory
# that contains the MORAI/ package root (one level above PROJECT_ROOT/..)
IMPORT_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
export PYTHONPATH="${IMPORT_ROOT}:${PYTHONPATH}"

# Configure Hugging Face to store cached models and datasets in the current directory's .cache folder
export HF_HOME="${PROJECT_ROOT}/.cache"

echo "=========================================================="
echo "Starting ECRD evaluation on TreeBench"
echo "Hugging Face Cache: ${HF_HOME}"
echo "=========================================================="
echo ""

# Run evaluation -- pass exactly one of --base / --supervisor / --ecrd
python "${PROJECT_ROOT}/scripts/test/eval_treebench.py" \
  "$@"


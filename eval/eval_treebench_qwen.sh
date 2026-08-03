#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Get the root directory of the project (one level up from the eval/ directory)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# eval_treebench_qwen.py imports "SSS.inference" / "SSS.fovea", so PYTHONPATH
# needs the directory that CONTAINS the SSS/ package (i.e. the MORAI/ root).
MORAI_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
export PYTHONPATH="${MORAI_ROOT}:${PYTHONPATH}"

# Configure Hugging Face to store cached models and datasets in the current directory's .cache folder
export HF_HOME="${PROJECT_ROOT}/.cache"

echo "=========================================================="
echo "Starting Qwen2.5-VL evaluation on TreeBench"
echo "=========================================================="
echo ""

# Run evaluation -- pass --method base or --method vdgd (required), plus any
# other eval_treebench_qwen.py flags (, --min-k, --max-k, --limit, ...)
python "${PROJECT_ROOT}/eval/eval_treebench_qwen.py" \
  "$@"




# bash eval/eval_treebench_qwen.sh --method base --limit 20
# bash eval/eval_treebench_qwen.sh --method vdgd
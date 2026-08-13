#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "=========================================================="
echo "Running All See-It-Say-It-Sorted Benchmarks"
echo "=========================================================="

# echo ""
# echo ">>> [1/9] Running V* Bench (Base)..."
# bash "${PROJECT_ROOT}/scripts/test/eval_vstar.sh" --base

# echo ""
# echo ">>> [2/9] Running V* Bench (+supervisor)..."
# bash "${PROJECT_ROOT}/scripts/test/eval_vstar.sh" --supervisor

# echo ""
# echo ">>> [3/9] Running V* Bench (+ECRD)..."
# bash "${PROJECT_ROOT}/scripts/test/eval_vstar.sh" --ecrd --grit-device 0

# echo ""
# echo ">>> [4/9] Running TreeBench (Base)..."
# bash "${PROJECT_ROOT}/scripts/test/eval_treebench.sh" --base

# echo ""
# echo ">>> [5/9] Running TreeBench (+supervisor)..."
# bash "${PROJECT_ROOT}/scripts/test/eval_treebench.sh" --supervisor

echo ""
echo ">>> [6/9] Running TreeBench (+ECRD)..."
bash "${PROJECT_ROOT}/scripts/test/eval_treebench.sh" --ecrd --grit-device 0

# echo ""
# echo ">>> [7/9] Running RH-Bench (Base)..."
# bash "${PROJECT_ROOT}/scripts/test/eval_rhbench.sh" --base

# echo ""
# echo ">>> [8/9] Running RH-Bench (+supervisor)..."
# bash "${PROJECT_ROOT}/scripts/test/eval_rhbench.sh" --supervisor

# echo ""
# echo ">>> [9/9] Running RH-Bench (+ECRD)..."
# bash "${PROJECT_ROOT}/scripts/test/eval_rhbench.sh" --ecrd --grit-device 0

echo ""
echo "=========================================================="
echo "All Benchmarks Completed Successfully!"
echo "=========================================================="

#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "=========================================================="
echo "Running All See-It-Say-It-Sorted Benchmarks"
echo "=========================================================="

echo ""
echo ">>> [1/6] Running V* Bench (Basic ECRD)..."
bash "${PROJECT_ROOT}/scripts/test/eval_vstar.sh" --limit 200 --model Qwen/Qwen2.5-VL-3B-Instruct

echo ""
echo ">>> [2/6] Running V* Bench (ECRD + GRIT)..."
bash "${PROJECT_ROOT}/scripts/test/eval_vstar.sh" --limit 200 --use-grit --grit-device 0 --model Qwen/Qwen2.5-VL-3B-Instruct

echo ""
echo ">>> [3/6] Running TreeBench (Basic ECRD)..."
bash "${PROJECT_ROOT}/scripts/test/eval_treebench.sh" --limit 500 --model Qwen/Qwen2.5-VL-3B-Instruct

echo ""
echo ">>> [4/6] Running TreeBench (ECRD + GRIT)..."
bash "${PROJECT_ROOT}/scripts/test/eval_treebench.sh" --limit 500 --use-grit --grit-device 0 --model Qwen/Qwen2.5-VL-3B-Instruct

echo ""
echo ">>> [5/6] Running RH-Bench (Basic ECRD)..."
bash "${PROJECT_ROOT}/scripts/test/eval_rhbench.sh" --limit 900 --model Qwen/Qwen2.5-VL-3B-Instruct

echo ""
echo ">>> [6/6] Running RH-Bench (ECRD + GRIT)..."
bash "${PROJECT_ROOT}/scripts/test/eval_rhbench.sh" --use-grit --grit-device 0 --limit 900 --model Qwen/Qwen2.5-VL-3B-Instruct

echo ""
echo "=========================================================="
echo "All Benchmarks Completed Successfully!"
echo "=========================================================="

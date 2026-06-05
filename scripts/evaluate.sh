#!/usr/bin/env bash
set -euo pipefail

# Run eval_only with just model/dataset/subset/seed.
# Usage:
#   bash scripts/evaluate.sh neumf Amazon_All_Beauty_18 sentiment_pos 42
#   DEVICE=1 bash scripts/evaluate.sh rgcl Amazon_Musical_Instruments_14 sentiment_neg 64
#
# Optional overrides:
#   CHECKPOINT_PATH=/path/to/model_best.pt bash scripts/evaluate.sh neumf Amazon_All_Beauty_18 sentiment_pos 42
#   OUTPUT_SEARCH_ROOT=./outputs EVAL_ROOT=./outputs/eval bash scripts/evaluate.sh neumf Amazon_All_Beauty_18

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL="${1:-}"
DATASET="${2:-}"
SUBSET="${3:-sentiment_pos}"
SEED="${4:-${SEED:-42}}"
DEVICE="${DEVICE:-0}"
OUTPUT_SEARCH_ROOT="${OUTPUT_SEARCH_ROOT:-./outputs}"
EVAL_ROOT="${EVAL_ROOT:-./outputs/eval/${MODEL}_${DATASET}_${SEED}_${SUBSET}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

if [[ -z "${MODEL}" || -z "${DATASET}" ]]; then
  echo "Usage: bash scripts/evaluate.sh <model> <dataset> [sentiment_pos|sentiment_neg|null] [seed]" >&2
  echo "Example: bash scripts/evaluate.sh neumf Amazon_All_Beauty_18 sentiment_pos 42" >&2
  exit 1
fi

find_checkpoint() {
  python - "${OUTPUT_SEARCH_ROOT}" "${MODEL}" "${DATASET}" "${SEED}" <<'PY'
import os
import sys

root, model, dataset, seed = sys.argv[1:]
checkpoint_name = f"{model}_best.pt"
prefix = f"{model}_{dataset}_{seed}_"
candidates = []

for current_root, _, files in os.walk(root):
    if checkpoint_name not in files:
        continue
    run_dir = os.path.basename(current_root)
    if run_dir.startswith(prefix):
        candidates.append(os.path.join(current_root, checkpoint_name))

if not candidates:
    sys.exit(1)

candidates.sort()
print(candidates[-1])
PY
}

latest_eval_dir() {
  python - "${EVAL_ROOT}" "${MODEL}" "${DATASET}" "${SEED}" <<'PY'
import os
import sys

root, model, dataset, seed = sys.argv[1:]
prefix = f"{model}_{dataset}_{seed}_"
candidates = []

if os.path.isdir(root):
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path) and name.startswith(prefix):
            candidates.append(path)

if not candidates:
    sys.exit(1)

candidates.sort()
print(candidates[-1])
PY
}

print_eval_result() {
  local result_path="$1"
  python - "${result_path}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    result = json.load(f)

metrics = result.get("test_metrics", {})
print("[evaluate] test_metrics")
for key in ("rmse", "mse", "mae"):
    if key in metrics:
        print(f"  {key}: {metrics[key]}")

sentiment = result.get("sentiment_metrics")
if sentiment:
    print("[evaluate] sentiment_metrics")
    for group, values in sentiment.items():
        print(f"  {group}:")
        for key in ("num_samples", "rmse", "mse", "mae"):
            if key in values:
                print(f"    {key}: {values[key]}")
PY
}

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  if ! CHECKPOINT_PATH="$(find_checkpoint)"; then
    echo "[evaluate] checkpoint not found under ${OUTPUT_SEARCH_ROOT}" >&2
    echo "[evaluate] expected pattern: ${MODEL}_${DATASET}_${SEED}_*/${MODEL}_best.pt" >&2
    echo "[evaluate] or pass CHECKPOINT_PATH=/path/to/${MODEL}_best.pt" >&2
    exit 1
  fi
fi

mkdir -p "${EVAL_ROOT}"

SUBSET_ARG="evaluation.sentiment_subset=${SUBSET}"
if [[ "${SUBSET}" == "null" || "${SUBSET}" == "none" || "${SUBSET}" == "None" ]]; then
  SUBSET_ARG="evaluation.sentiment_subset=null"
fi

echo "[evaluate] model=${MODEL} dataset=${DATASET} seed=${SEED} subset=${SUBSET} device=${DEVICE}"
echo "[evaluate] checkpoint=${CHECKPOINT_PATH}"
echo "[evaluate] eval_root=${EVAL_ROOT}"

python run.py \
  model="${MODEL}" \
  data.dataset="${DATASET}" \
  experiment.seed="${SEED}" \
  experiment.device="${DEVICE}" \
  experiment.save_dir="${EVAL_ROOT}" \
  evaluation.eval_only=true \
  evaluation.checkpoint_path="${CHECKPOINT_PATH}" \
  "${SUBSET_ARG}"

EVAL_DIR="$(latest_eval_dir)"
RESULT_PATH="${EVAL_DIR}/eval_only_results.json"

echo "[evaluate] result=${RESULT_PATH}"
print_eval_result "${RESULT_PATH}"

#!/usr/bin/env bash
set -euo pipefail

# Run from repository root regardless of where this script is launched.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DEVICE="${DEVICE:-0}"

# Change this to group runs under a custom folder, e.g.:
#   RUN_SET=align01 bash scripts/mymodel_2.sh
RUN_SET="${RUN_SET:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/mymodel/${RUN_SET}}"

mkdir -p "${OUTPUT_ROOT}"

run_named() {
  local name="$1"
  shift

  echo "[mymodel] name=${name} device=${DEVICE} output=${OUTPUT_ROOT}/${name}"
  python run.py \
    "$@" \
    experiment.device="${DEVICE}" \
    experiment.save_dir="${OUTPUT_ROOT}/${name}"
}

# Usage:
#   run_named <name-you-want> <normal hydra args...>
#
# The actual run directory becomes:
#   outputs/mymodel/<RUN_SET>/<name-you-want>/<model_dataset_seed_timestamp>/
#
# Copy/paste template:
#   run_named align01 \
#     model=mymodel_v5 \
#     data.dataset=Amazon_Office_Products_14 \
#     model.dropout=0.8 \
#     model.lambda_pair_align=1 \
#     model.orthogonal_residual_weight=0.1 \
#     experiment.seed=42 \
#     training.batch=64
#
# Keep one space before each trailing backslash: `experiment.seed=42 \`

run_named off_res01 \
  model=mymodel_v5 \
  data.dataset=Amazon_Office_Products_14 \
  model.dropout=0.8 \
  model.lambda_pair_align=1 \
  model.orthogonal_residual_weight=0.1 \
  experiment.seed=42 \
  training.batch=64
run_named off_res02 \
  model=mymodel_v5 \
  data.dataset=Amazon_Office_Products_14 \
  model.dropout=0.8 \
  model.lambda_pair_align=1 \
  model.orthogonal_residual_weight=0.2 \
  experiment.seed=42 \
  training.batch=64
run_named off_res03 \
  model=mymodel_v5 \
  data.dataset=Amazon_Office_Products_14 \
  model.dropout=0.8 \
  model.lambda_pair_align=1 \
  model.orthogonal_residual_weight=0.3 \
  experiment.seed=42 \
  training.batch=64
run_named off_res04 \
  model=mymodel_v5 \
  data.dataset=Amazon_Office_Products_14 \
  model.dropout=0.8 \
  model.lambda_pair_align=1 \
  model.orthogonal_residual_weight=0.4 \
  experiment.seed=42 \
  training.batch=64
run_named off_res05 \
  model=mymodel_v5 \
  data.dataset=Amazon_Office_Products_14 \
  model.dropout=0.8 \
  model.lambda_pair_align=1 \
  model.orthogonal_residual_weight=0.5 \
  experiment.seed=42 \
  training.batch=64

run_named off_res06 \
  model=mymodel_v5 \
  data.dataset=Amazon_Office_Products_14 \
  model.dropout=0.8 \
  model.lambda_pair_align=1 \
  model.orthogonal_residual_weight=0.6 \
  experiment.seed=42 \
  training.batch=64

run_named off_res07 \
  model=mymodel_v5 \
  data.dataset=Amazon_Office_Products_14 \
  model.dropout=0.8 \
  model.lambda_pair_align=1 \
  model.orthogonal_residual_weight=0.7 \
  experiment.seed=42 \
  training.batch=64

run_named off_res08 \
  model=mymodel_v5 \
  data.dataset=Amazon_Office_Products_14 \
  model.dropout=0.8 \
  model.lambda_pair_align=1 \
  model.orthogonal_residual_weight=0.8 \
  experiment.seed=42 \
  training.batch=64

run_named off_res09 \
  model=mymodel_v5 \
  data.dataset=Amazon_Office_Products_14 \
  model.dropout=0.8 \
  model.lambda_pair_align=1 \
  model.orthogonal_residual_weight=0.9 \
  experiment.seed=42 \
  training.batch=64

run_named off_res10 \
  model=mymodel_v5 \
  data.dataset=Amazon_Office_Products_14 \
  model.dropout=0.8 \
  model.lambda_pair_align=1 \
  model.orthogonal_residual_weight=1 \
  experiment.seed=42 \
  training.batch=64
echo "[mymodel] all runs saved under: ${OUTPUT_ROOT}"

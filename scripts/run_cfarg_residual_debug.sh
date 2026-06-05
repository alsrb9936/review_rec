#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-0}"
DATASET="${DATASET:-Amazon_Musical_Instruments_14}"
SEED="${SEED:-42}"
RESULT_DIR="./results/cfarg_residual_debug"
mkdir -p "${RESULT_DIR}"

COMMON_OVERRIDES=(
  "experiment.device=${DEVICE}"
  "experiment.results_dir=${RESULT_DIR}"
  "training.batch=128"
  "training.eval_batch=512"
  "training.epoch=100"
  "training.lr=0.001"
  "training.weight_decay=1e-4"
  "training.grad_clip=1.0"
  "data.exclude_target_review=true"
)

python scripts/sync_cfarg_bert_splits_from_common.py --dataset "${DATASET}"

python run.py model=cfarg_cf_only data.dataset="${DATASET}" experiment.seed="${SEED}" \
  experiment.result_file="${RESULT_DIR}/main_results.csv" "${COMMON_OVERRIDES[@]}"

for scale in 0.001 0.003 0.01; do
  python run.py model=cfarg_fusion data.dataset="${DATASET}" experiment.seed="${SEED}" \
    model.review_scale_init="${scale}" \
    experiment.result_file="${RESULT_DIR}/main_results.csv" "${COMMON_OVERRIDES[@]}"

  python run.py model=cfarg_gate_control data.dataset="${DATASET}" experiment.seed="${SEED}" \
    model.review_scale_init="${scale}" \
    model.gate_reg_weight=0.01 \
    model.align_threshold=0.2 \
    experiment.result_file="${RESULT_DIR}/main_results.csv" "${COMMON_OVERRIDES[@]}"
done

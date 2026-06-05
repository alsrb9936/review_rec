#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-1}"
DATASETS=(
  "Amazon_Musical_Instruments_14"
  "Amazon_Office_Products_14"
  "Amazon_Digital_Music_14"
)
SEEDS=(42 43 44)

RESULT_DIR="./results/cfarg_gate_control_noise"
mkdir -p "${RESULT_DIR}"

COMMON_OVERRIDES=(
  "experiment.device=${DEVICE}"
  "experiment.results_dir=${RESULT_DIR}"
  "experiment.result_file=${RESULT_DIR}/main_results.csv"
  "training.batch=128"
  "training.eval_batch=512"
  "training.epoch=100"
  "training.lr=0.001"
  "training.weight_decay=1e-4"
  "training.grad_clip=1.0"
  "noise.enabled=true"
  "noise.ratio=0.3"
)

for dataset in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    python run.py model=cfarg_gate_control data.dataset="${dataset}" experiment.seed="${seed}" "${COMMON_OVERRIDES[@]}"
  done
done

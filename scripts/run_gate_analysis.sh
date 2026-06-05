#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-Amazon_All_Beauty_18}"
SEEDS="${SEEDS:-42 43 44}"
DEVICE="${DEVICE:-0}"
FAST_DEV="${FAST_DEV:-false}"

mkdir -p results
mkdir -p results/cfarg_fixed

for seed in ${SEEDS}; do
  python run.py --config-name cfarg data.dataset="${DATASET}" experiment.seed="${seed}" experiment.device="${DEVICE}" experiment.fast_dev_run="${FAST_DEV}" experiment.results_dir=./results/cfarg_fixed experiment.result_file=./results/cfarg_fixed/gate_stats.csv
done

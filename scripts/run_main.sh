#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-Amazon_Office_Products_14}"
SEEDS="${SEEDS:-42 43 44}"
DEVICE="${DEVICE:-1}"
FAST_DEV="${FAST_DEV:-false}"

mkdir -p results
mkdir -p results/cfarg_fixed

for seed in ${SEEDS}; do
  python run.py --config-name cf_only data.dataset="${DATASET}" experiment.seed="${seed}" experiment.device="${DEVICE}" experiment.fast_dev_run="${FAST_DEV}" experiment.results_dir=./results/cfarg_fixed experiment.result_file=./results/cfarg_fixed/main_results.csv
  python run.py --config-name whole_review_fusion data.dataset="${DATASET}" experiment.seed="${seed}" experiment.device="${DEVICE}" experiment.fast_dev_run="${FAST_DEV}" experiment.results_dir=./results/cfarg_fixed experiment.result_file=./results/cfarg_fixed/main_results.csv
  python run.py --config-name cfarg data.dataset="${DATASET}" experiment.seed="${seed}" experiment.device="${DEVICE}" experiment.fast_dev_run="${FAST_DEV}" experiment.results_dir=./results/cfarg_fixed experiment.result_file=./results/cfarg_fixed/main_results.csv
done

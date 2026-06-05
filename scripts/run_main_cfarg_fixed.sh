#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-Amazon_Digital_Music_14}"
SEEDS="${SEEDS:-42 43 44}"
DEVICE="${DEVICE:-2}"
FAST_DEV="${FAST_DEV:-false}"
RESULTS_DIR="${RESULTS_DIR:-./results/cfarg_fixed}"

mkdir -p "${RESULTS_DIR}"

for seed in ${SEEDS}; do
  python run.py --config-name cf_only data.dataset="${DATASET}" experiment.seed="${seed}" experiment.device="${DEVICE}" experiment.fast_dev_run="${FAST_DEV}" experiment.results_dir="${RESULTS_DIR}" experiment.result_file="${RESULTS_DIR}/main_results.csv"
  python run.py --config-name whole_review_fusion data.dataset="${DATASET}" experiment.seed="${seed}" experiment.device="${DEVICE}" experiment.fast_dev_run="${FAST_DEV}" experiment.results_dir="${RESULTS_DIR}" experiment.result_file="${RESULTS_DIR}/main_results.csv"
  python run.py --config-name cfarg data.dataset="${DATASET}" experiment.seed="${seed}" experiment.device="${DEVICE}" experiment.fast_dev_run="${FAST_DEV}" experiment.results_dir="${RESULTS_DIR}" experiment.result_file="${RESULTS_DIR}/main_results.csv"
done

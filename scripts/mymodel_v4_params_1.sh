#!/usr/bin/env bash
set -euo pipefail

# Run from repository root regardless of where the script is launched.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Override from the shell when needed, e.g.:
#   DEVICE=1 SEEDS="42 64 57" DATASETS="Amazon_Musical_Instruments_14" bash scripts/mymodel_v4_params.sh
DEVICE="${DEVICE:-0}"
SEEDS="${SEEDS:-42}"
DATASETS="${DATASETS:-Amazon_Musical_Instruments_14}"

# Shared budget. Override these for shorter smoke tests or full runs.
EPOCH="${EPOCH:-200}"
PATIENCE="${PATIENCE:-10}"
BATCH="${BATCH:-64}"
EVAL_BATCH="${EVAL_BATCH:-512}"

# All runs from this script are grouped here, then summarized into summary.csv.
SWEEP_NAME="${SWEEP_NAME:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_DIR="${SWEEP_DIR:-./outputs/mymodel_v4_params_1/${SWEEP_NAME}}"
MANIFEST_PATH="${SWEEP_DIR}/manifest.tsv"
SUMMARY_PATH="${SWEEP_DIR}/summary.csv"

# Extra Hydra overrides appended to every run, e.g.:
#   EXTRA_OVERRIDES="training.epoch=20 evaluation.early_stop_patience=3" bash scripts/mymodel_v4_params.sh
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

mkdir -p "${SWEEP_DIR}"
printf 'tag\tdataset\tseed\toverrides\toutput_dir\n' > "${MANIFEST_PATH}"

latest_output_dir() {
  local case_save_root="$1"
  local dataset="$2"
  local seed="$3"
  local latest=""

  for candidate in "${case_save_root}"/mymodel_v4_"${dataset}"_"${seed}"_*; do
    [[ -d "${candidate}" ]] || continue
    if [[ -z "${latest}" || "${candidate}" > "${latest}" ]]; then
      latest="${candidate}"
    fi
  done

  printf '%s\n' "${latest}"
}

run_case() {
  local tag="$1"
  local dataset="$2"
  local seed="$3"
  local overrides="$4"
  local case_save_root="${SWEEP_DIR}/${tag}"
  local output_dir=""
  shift 4

  mkdir -p "${case_save_root}"

  echo "[mymodel_v4 params] tag=${tag} dataset=${dataset} seed=${seed} device=${DEVICE}"
  python run.py \
    model=mymodel_v4 \
    data.dataset="${dataset}" \
    experiment.seed="${seed}" \
    experiment.device="${DEVICE}" \
    training.epoch="${EPOCH}" \
    evaluation.early_stop_patience="${PATIENCE}" \
    training.batch="${BATCH}" \
    training.eval_batch="${EVAL_BATCH}" \
    experiment.save_dir="${case_save_root}" \
    "$@" \
    ${EXTRA_OVERRIDES}

  output_dir="$(latest_output_dir "${case_save_root}" "${dataset}" "${seed}")"
  printf '%s\t%s\t%s\t%s\t%s\n' "${tag}" "${dataset}" "${seed}" "${overrides} ${EXTRA_OVERRIDES}" "${output_dir}" >> "${MANIFEST_PATH}"
}

write_summary() {
  python - "${MANIFEST_PATH}" "${SUMMARY_PATH}" <<'PY'
import csv
import json
import os
import sys

manifest_path, summary_path = sys.argv[1], sys.argv[2]

rows = []
param_keys = set()

with open(manifest_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        overrides = row.get("overrides", "").strip()
        params = {}
        for token in overrides.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            params[key] = value
            param_keys.add(key)

        output_dir = row.get("output_dir", "")
        result_path = os.path.join(output_dir, "test_results.json") if output_dir else ""
        status = "ok"
        result = {}
        if not result_path or not os.path.exists(result_path):
            status = "missing_result"
        else:
            with open(result_path, "r", encoding="utf-8") as rf:
                result = json.load(rf)

        metrics = result.get("test_metrics", {}) if isinstance(result, dict) else {}
        rows.append(
            {
                "tag": row.get("tag", ""),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "status": status,
                "best_valid_metric": result.get("best_valid_metric", "") if isinstance(result, dict) else "",
                "best_valid_metric_name": result.get("best_valid_metric_name", "") if isinstance(result, dict) else "",
                "test_rmse": metrics.get("rmse", ""),
                "test_mse": metrics.get("mse", ""),
                "test_mae": metrics.get("mae", ""),
                "overrides": overrides,
                "output_dir": output_dir,
                **{f"param:{key}": value for key, value in params.items()},
            }
        )

param_columns = [f"param:{key}" for key in sorted(param_keys)]
columns = [
    "tag",
    "dataset",
    "seed",
    "status",
    "best_valid_metric",
    "best_valid_metric_name",
    "test_rmse",
    "test_mse",
    "test_mae",
    *param_columns,
    "overrides",
    "output_dir",
]

os.makedirs(os.path.dirname(summary_path), exist_ok=True)
with open(summary_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

print(f"[mymodel_v4 params] summary saved: {summary_path}")
PY
}

# Format: tag::Hydra overrides
# Add/remove rows here to test more model settings.
CASES=(
  "baseline::"
  "pair_align_0.01::model.lambda_pair_align=0.01"
  "pair_align_0.05::model.lambda_pair_align=0.05"
  "temp_0.1::model.temperature=0.1"
  "temp_0.5::model.temperature=0.5"
  "residual_0.5::model.orthogonal_residual_weight=0.5"
  "residual_1.0::model.orthogonal_residual_weight=1.0"
  "dropout_0.3::model.dropout=0.3"
  "dropout_0.8::model.dropout=0.8"
  "lr_5e-4::training.lr=0.0005"
  "wd_1e-5::training.weight_decay=1e-5"
  "align_residual::model.lambda_pair_align=0.01 model.orthogonal_residual_weight=0.5"
)

for dataset in ${DATASETS}; do
  for seed in ${SEEDS}; do
    for case_spec in "${CASES[@]}"; do
      tag="${case_spec%%::*}"
      overrides="${case_spec#*::}"

      # shellcheck disable=SC2086 # Intentionally split Hydra override list.
      run_case "${tag}" "${dataset}" "${seed}" "${overrides}" ${overrides}
    done
  done
done

write_summary
echo "[mymodel_v4 params] manifest: ${MANIFEST_PATH}"
echo "[mymodel_v4 params] summary:  ${SUMMARY_PATH}"

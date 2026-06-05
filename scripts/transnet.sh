python run.py \
  model=neumf \
  data.dataset="${DATASET}" \
  experiment.seed="${SEED}" \
  experiment.device="${DEVICE}" \
  experiment.save_dir="${EVAL_ROOT}" \
  evaluation.eval_only=true \
  evaluation.checkpoint_path="${CHECKPOINT_PATH}" \
  "${SUBSET_ARG}"

DEVICE=0


python run.py model=mymodel_v5 data.dataset=Amazon_Digital_Music_14 experiment.seed=42 experiment.device=$DEVICE training.batch=128 model.lambda_pair_align=1 model.orthogonal_residual_weight=0.0
python run.py model=mymodel_v5 data.dataset=Amazon_Digital_Music_14 experiment.seed=64 experiment.device=$DEVICE training.batch=128 model.lambda_pair_align=1 model.orthogonal_residual_weight=0.0
python run.py model=mymodel_v5 data.dataset=Amazon_Digital_Music_14 experiment.seed=57 experiment.device=$DEVICE training.batch=128 model.lambda_pair_align=1 model.orthogonal_residual_weight=0.0
python run.py model=mymodel_v5 data.dataset=Amazon_Digital_Music_14 experiment.seed=2025 experiment.device=$DEVICE training.batch=128 model.lambda_pair_align=1 model.orthogonal_residual_weight=0.0
python run.py model=mymodel_v5 data.dataset=Amazon_Digital_Music_14 experiment.seed=2026 experiment.device=$DEVICE training.batch=128 model.lambda_pair_align=1 model.orthogonal_residual_weight=0.0
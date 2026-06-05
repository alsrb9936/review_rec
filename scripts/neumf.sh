cd ../
DEVICE=1

python run.py model=neumf data.dataset=Amazon_Musical_Instruments_14 experiment.seed=42 experiment.device=$DEVICE
python run.py model=neumf data.dataset=Amazon_Musical_Instruments_14 experiment.seed=64 experiment.device=$DEVICE
python run.py model=neumf data.dataset=Amazon_Musical_Instruments_14 experiment.seed=57 experiment.device=$DEVICE
python run.py model=neumf data.dataset=Amazon_Musical_Instruments_14 experiment.seed=2025 experiment.device=$DEVICE
python run.py model=neumf data.dataset=Amazon_Musical_Instruments_14 experiment.seed=2026 experiment.device=$DEVICE

python run.py model=neumf data.dataset=Amazon_Office_Products_14 experiment.seed=42 experiment.device=$DEVICE training.batch=128
python run.py model=neumf data.dataset=Amazon_Office_Products_14 experiment.seed=64 experiment.device=$DEVICE training.batch=128
python run.py model=neumf data.dataset=Amazon_Office_Products_14 experiment.seed=57 experiment.device=$DEVICE training.batch=128
python run.py model=neumf data.dataset=Amazon_Office_Products_14 experiment.seed=2025 experiment.device=$DEVICE training.batch=128
python run.py model=neumf data.dataset=Amazon_Office_Products_14 experiment.seed=2026 experiment.device=$DEVICE training.batch=128

python run.py model=neumf data.dataset=Amazon_Digital_Music_14 experiment.seed=42 experiment.device=$DEVICE training.batch=128
python run.py model=neumf data.dataset=Amazon_Digital_Music_14 experiment.seed=64 experiment.device=$DEVICE training.batch=128
python run.py model=neumf data.dataset=Amazon_Digital_Music_14 experiment.seed=57 experiment.device=$DEVICE training.batch=128
python run.py model=neumf data.dataset=Amazon_Digital_Music_14 experiment.seed=2025 experiment.device=$DEVICE training.batch=128
python run.py model=neumf data.dataset=Amazon_Digital_Music_14 experiment.seed=2026 experiment.device=$DEVICE training.batch=128


python run.py model=neumf \
  data.dataset=Amazon_All_Beauty_18 \
  evaluation.eval_only=true \
  evaluation.checkpoint_path=/home/infolab/mnt/mingyu/review_rec/review_reproducibility/outputs/neumf_Amazon_All_Beauty_18_42_20260604_115308/neumf_best.pt \
  evaluation.sentiment_subset=sentiment_pos

# python preprocess.py data.dataset=Amazon_Digital_Music_14 data.type=sentiment experiment.device=0
# python preprocess.py data.dataset=Amazon_Office_Products_14 data.type=sentiment experiment.device=1
# python preprocess.py data.dataset=Amazon_Musical_Instruments_14 data.type=sentiment experiment.device=2

python run.py model=mymodel_v4 \
  data.dataset=Amazon_Musical_Instruments_14 \
  model.dropout=0.8 \
  model.lambda_pair_align=1 \
  model.orthogonal_residual_weight=0.1 \
  experiment.seed=42


python run.py model=neumf \
  data.dataset=Amazon_Musical_Instruments_14 \
  evaluation.eval_only=true \
  evaluation.checkpoint_path=/home/infolab/mnt/mingyu/review_rec/review_reproducibility/outputs/neumf_Amazon_Musical_Instruments_14_42_20260604_171021/neumf_best.pt \
  evaluation.sentiment_subset=sentiment_pos
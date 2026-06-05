cd ../

DEVICE=2

python run.py model=rmg data.dataset=Amazon_Musical_Instruments_14 experiment.seed=42 experiment.device=$DEVICE
python run.py model=rmg data.dataset=Amazon_Musical_Instruments_14 experiment.seed=64 experiment.device=$DEVICE
python run.py model=rmg data.dataset=Amazon_Musical_Instruments_14 experiment.seed=57 experiment.device=$DEVICE
python run.py model=rmg data.dataset=Amazon_Musical_Instruments_14 experiment.seed=2025 experiment.device=$DEVICE
python run.py model=rmg data.dataset=Amazon_Musical_Instruments_14 experiment.seed=2026 experiment.device=$DEVICE

python run.py model=rmg data.dataset=Amazon_Office_Products_14 experiment.seed=42 experiment.device=$DEVICE training.batch=128
python run.py model=rmg data.dataset=Amazon_Office_Products_14 experiment.seed=64 experiment.device=$DEVICE training.batch=128
python run.py model=rmg data.dataset=Amazon_Office_Products_14 experiment.seed=57 experiment.device=$DEVICE training.batch=128
python run.py model=rmg data.dataset=Amazon_Office_Products_14 experiment.seed=2025 experiment.device=$DEVICE training.batch=128
python run.py model=rmg data.dataset=Amazon_Office_Products_14 experiment.seed=2026 experiment.device=$DEVICE training.batch=128

python run.py model=rmg data.dataset=Amazon_Digital_Music_14 experiment.seed=42 experiment.device=$DEVICE training.batch=128
python run.py model=rmg data.dataset=Amazon_Digital_Music_14 experiment.seed=64 experiment.device=$DEVICE training.batch=128
python run.py model=rmg data.dataset=Amazon_Digital_Music_14 experiment.seed=57 experiment.device=$DEVICE training.batch=128
python run.py model=rmg data.dataset=Amazon_Digital_Music_14 experiment.seed=2025 experiment.device=$DEVICE training.batch=128
python run.py model=rmg data.dataset=Amazon_Digital_Music_14 experiment.seed=2026 experiment.device=$DEVICE training.batch=128

python analysis.py \
  --checkpoint /home/infolab/mnt/mingyu/review_rec/review_reproducibility/outputs/mymodel_v4_Amazon_Digital_Music_14_64_20260604_172039/mymodel_v4_best.pt \
  --dataset Amazon_Digital_Music_14 \
  --subset sentiment_pos \
  --device 0
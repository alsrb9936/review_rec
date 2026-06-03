cd ../

DEVICE=3

# python run.py model=transnet data.dataset=Amazon_Musical_Instruments_14 experiment.seed=42 experiment.device=$DEVICE
# python run.py model=transnet data.dataset=Amazon_Musical_Instruments_14 experiment.seed=64 experiment.device=$DEVICE
# python run.py model=transnet data.dataset=Amazon_Musical_Instruments_14 experiment.seed=57 experiment.device=$DEVICE
# python run.py model=transnet data.dataset=Amazon_Musical_Instruments_14 experiment.seed=2025 experiment.device=$DEVICE
# python run.py model=transnet data.dataset=Amazon_Musical_Instruments_14 experiment.seed=2026 experiment.device=$DEVICE

python run.py model=transnet data.dataset=Amazon_Office_Products_14 experiment.seed=42 experiment.device=$DEVICE training.batch=128
python run.py model=transnet data.dataset=Amazon_Office_Products_14 experiment.seed=64 experiment.device=$DEVICE training.batch=128
python run.py model=transnet data.dataset=Amazon_Office_Products_14 experiment.seed=57 experiment.device=$DEVICE training.batch=128
python run.py model=transnet data.dataset=Amazon_Office_Products_14 experiment.seed=2025 experiment.device=$DEVICE training.batch=128
python run.py model=transnet data.dataset=Amazon_Office_Products_14 experiment.seed=2026 experiment.device=$DEVICE training.batch=128

python run.py model=transnet data.dataset=Amazon_Digital_Music_14 experiment.seed=42 experiment.device=$DEVICE training.batch=128
python run.py model=transnet data.dataset=Amazon_Digital_Music_14 experiment.seed=64 experiment.device=$DEVICE training.batch=128
python run.py model=transnet data.dataset=Amazon_Digital_Music_14 experiment.seed=57 experiment.device=$DEVICE training.batch=128
python run.py model=transnet data.dataset=Amazon_Digital_Music_14 experiment.seed=2025 experiment.device=$DEVICE training.batch=128
python run.py model=transnet data.dataset=Amazon_Digital_Music_14 experiment.seed=2026 experiment.device=$DEVICE training.batch=128
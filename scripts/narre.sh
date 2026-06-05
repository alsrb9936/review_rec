cd ../

DEVICE=1

python run.py model=narre data.dataset=Amazon_Musical_Instruments_14 experiment.seed=42 experiment.device=$DEVICE
python run.py model=narre data.dataset=Amazon_Musical_Instruments_14 experiment.seed=64 experiment.device=$DEVICE
python run.py model=narre data.dataset=Amazon_Musical_Instruments_14 experiment.seed=57 experiment.device=$DEVICE
python run.py model=narre data.dataset=Amazon_Musical_Instruments_14 experiment.seed=2025 experiment.device=$DEVICE
python run.py model=narre data.dataset=Amazon_Musical_Instruments_14 experiment.seed=2026 experiment.device=$DEVICE

python run.py model=narre data.dataset=Amazon_Office_Products_14 experiment.seed=42 experiment.device=1 training.batch=128
python run.py model=narre data.dataset=Amazon_Office_Products_14 experiment.seed=64 experiment.device=0 training.batch=128
python run.py model=narre data.dataset=Amazon_Office_Products_14 experiment.seed=57 experiment.device=0 training.batch=128
python run.py model=narre data.dataset=Amazon_Office_Products_14 experiment.seed=2025 experiment.device=0 training.batch=128
python run.py model=narre data.dataset=Amazon_Office_Products_14 experiment.seed=2026 experiment.device=0 training.batch=128

python run.py model=narre data.dataset=Amazon_Digital_Music_14 experiment.seed=42 experiment.device=$DEVICE training.batch=128
python run.py model=narre data.dataset=Amazon_Digital_Music_14 experiment.seed=64 experiment.device=$DEVICE training.batch=128
python run.py model=narre data.dataset=Amazon_Digital_Music_14 experiment.seed=57 experiment.device=$DEVICE training.batch=128
python run.py model=narre data.dataset=Amazon_Digital_Music_14 experiment.seed=2025 experiment.device=$DEVICE training.batch=128
python run.py model=narre data.dataset=Amazon_Digital_Music_14 experiment.seed=2026 experiment.device=$DEVICE training.batch=128
python run.py model=neumf data.dataset=Amazon_Musical_Instruments_14 experiment.seed=64 experiment.device=0
python run.py model=transnet data.dataset=Amazon_Musical_Instruments_14 experiment.seed=64 experiment.device=0

python run.py model=deepconn data.dataset=Amazon_Musical_Instruments_14 experiment.seed=64 experiment.device=1
python run.py model=narre data.dataset=Amazon_Musical_Instruments_14 experiment.seed=64 experiment.device=2
python run.py model=daml data.dataset=Amazon_Musical_Instruments_14 experiment.seed=64 experiment.device=3


cd ../
DEVICE=0

python run.py model=recafr data.dataset=Amazon_Musical_Instruments_14 experiment.seed=42 experiment.device=$DEVICE

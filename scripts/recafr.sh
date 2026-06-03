cd ../
DEVICE=0

python run.py model=recafr data.dataset=Amazon_Office_Products_14 experiment.seed=2023 experiment.device=$DEVICE

# default만
python main.py
# model config 적용
python main.py --model neumf
# model + argparse로 추가 override
python main.py --model sasrec batch=64 lr=0.05 epoch=50
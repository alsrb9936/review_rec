# base config만 확인
python main.py --cfg job

# NeuMF 실행
python main.py model=neumf

# DeepCoNN 실행
python main.py model=deepconn

# model config + 추가 override
python main.py model=neumf training.batch=64 training.lr=0.05 training.epoch=50

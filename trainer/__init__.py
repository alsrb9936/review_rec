from trainer.base_trainer import BaseTrainer
from trainer.neumf_trainer import NeuMFTrainer
from trainer.deepconn_trainer import DeepCoNNTrainer

MODEL_TRAINER_DICT = {
    "neumf": NeuMFTrainer,
    "deepconn": DeepCoNNTrainer,
}
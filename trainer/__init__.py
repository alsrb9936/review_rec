from trainer.base_trainer import BaseTrainer
from trainer.neumf_trainer import NeuMFTrainer
from trainer.deepconn_trainer import DeepCoNNTrainer
from trainer.narre_trainer import NARRETrainer
from trainer.mymodel_trainer import MyModelTrainer
from trainer.rgcl_trainer import RGCLTrainer
from trainer.letter_trainer import LetterTrainer

MODEL_TRAINER_DICT = {
    "neumf": NeuMFTrainer,
    "deepconn": DeepCoNNTrainer,
    "narre": NARRETrainer,
    "mymodel": MyModelTrainer,
    "rgcl": RGCLTrainer,
    "mymodel_cfonly": MyModelTrainer,
    "mymodel_neumf": MyModelTrainer,
    "mymodel_v2": NARRETrainer,
    "letter": LetterTrainer,
}
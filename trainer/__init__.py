from trainer.base_trainer import BaseTrainer
from trainer.neumf_trainer import NeuMFTrainer
from trainer.deepconn_trainer import DeepCoNNTrainer
from trainer.narre_trainer import NARRETrainer
from trainer.mymodel_trainer import MyModelTrainer
MODEL_TRAINER_DICT = {
    "neumf": NeuMFTrainer,
    "deepconn": DeepCoNNTrainer,
    "narre": NARRETrainer,
    "mymodel": MyModelTrainer,
    "mymodel_cfonly": MyModelTrainer,
    "mymodel_concat": MyModelTrainer,
    "mymodel_shared": MyModelTrainer,
    "mymodel_full": MyModelTrainer,
}
from trainer.base_trainer import BaseTrainer
from trainer.neumf_trainer import NeuMFTrainer
from trainer.deepconn_trainer import DeepCoNNTrainer
from trainer.narre_trainer import NARRETrainer
from trainer.rgcl_trainer import RGCLTrainer
from trainer.daml_trainer import DAMLTrainer
from trainer.lightgcn_trainer import LightGCNTrainer
from trainer.transnet_trainer import TransNetTrainer
from trainer.mymodel_v1_trainer import MyModelV1Trainer
from trainer.letter_trainer import LETTERTrainer
from trainer.recafr_trainer import RecAFRTrainer
from trainer.rmg_trainer import RMGTrainer

MODEL_TRAINER_DICT = {
    "neumf": NeuMFTrainer,
    "deepconn": DeepCoNNTrainer,
    "narre": NARRETrainer,
    "rgcl": RGCLTrainer,
    "daml": DAMLTrainer,
    "lightgcn": LightGCNTrainer,
    "transnet": TransNetTrainer,
    "mymodel_v1": MyModelV1Trainer,
    "letter": LETTERTrainer,
    "recafr": RecAFRTrainer,
    "rmg": RMGTrainer,
}

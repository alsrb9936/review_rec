from trainer.base_trainer import BaseTrainer
from trainer.neumf_trainer import NeuMFTrainer
from trainer.deepconn_trainer import DeepCoNNTrainer
from trainer.narre_trainer import NARRETrainer
from trainer.rgcl_trainer import RGCLTrainer
from trainer.daml_trainer import DAMLTrainer
from trainer.lightgcn_trainer import LightGCNTrainer

MODEL_TRAINER_DICT = {
    "neumf": NeuMFTrainer,
    "deepconn": DeepCoNNTrainer,
    "narre": NARRETrainer,
    "rgcl": RGCLTrainer,
    "daml": DAMLTrainer,
    "lightgcn": LightGCNTrainer,
}

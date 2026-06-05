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
from trainer.mymodel_v2_trainer import MyModelV2Trainer
from trainer.mymodel_v3_trainer import MyModelV3Trainer
from trainer.mymodel_v4_trainer import MyModelV4Trainer
from trainer.mymodel_v5_trainer import MyModelV5Trainer
from trainer.cfarg_trainer import CFARGTrainer

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
    "mymodel_v2": MyModelV2Trainer,
    "mymodel_v3": MyModelV3Trainer,
    "mymodel_v4": MyModelV4Trainer,
    "mymodel_v5": MyModelV5Trainer,
    "cfarg": CFARGTrainer,
}

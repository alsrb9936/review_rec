from models.neumf import NeuMF
from models.deepconn import DeepCoNN
from models.narre import NARRE
from models.rgcl import RGCL
from models.daml import DAML
from models.lightgcn import LightGCN
from models.transnet import TransNet
from models.rgcl import RGCL
from models.mymodel_v1 import MyModelV1
from models.letter import LETTER
from models.recafr import RecAFR
from models.rmg import RMG
from models.mymodel_v2 import MyModelV2
from models.mymodel_v3 import MyModelV3
from models.mymodel_v4 import MyModelV4
from models.mymodel_v5 import MyModelV5

MODEL_DICT = {
    "neumf": NeuMF,
    "deepconn": DeepCoNN,
    "narre": NARRE,
    "rgcl": RGCL,
    "daml": DAML,
    "lightgcn": LightGCN,
    "transnet": TransNet,
    "mymodel_v1": MyModelV1,
    "letter": LETTER,
    "recafr": RecAFR,
    "rmg": RMG,
    "mymodel_v2": MyModelV2,
    "mymodel_v3": MyModelV3,
    "mymodel_v4": MyModelV4,
    "mymodel_v5": MyModelV5,
}

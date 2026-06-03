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
}

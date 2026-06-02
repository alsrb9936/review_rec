from models.neumf import NeuMF
from models.deepconn import DeepCoNN
from models.narre import NARRE
from models.rgcl import RGCL
from models.daml import DAML
from models.lightgcn import LightGCN
from models.recafr import ReCAFR
MODEL_DICT = {
    "neumf": NeuMF,
    "deepconn": DeepCoNN,
    "narre": NARRE,
    "rgcl": RGCL,
    "daml": DAML,
    "recafr" : ReCAFR,
    "lightgcn": LightGCN,
}

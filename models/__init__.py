from models.base_model import BaseModel
from models.neumf import NeuMF
from models.deepconn import DeepCoNN
from models.narre import NARRE
from models.mymodel import MyModel
from models.rgcl import RGCL

MODEL_DICT = {
    "neumf": NeuMF,
    "deepconn": DeepCoNN,
    "narre": NARRE,
    "mymodel": MyModel,
    "rgcl": RGCL,
}
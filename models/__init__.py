from models.base_model import BaseModel
from models.neumf import NeuMF
from models.deepconn import DeepCoNN
from models.narre import NARRE
from models.mymodel import MyModel
from models.rgcl import RGCL
from models.mymodel_cfonly import MyModelCFOnly
from models.mymodel_neumf import MyModelNueMF
from models.mymodel_v2 import MyModelV2
from models.letter import LetterModel
MODEL_DICT = {
    "neumf": NeuMF,
    "deepconn": DeepCoNN,
    "narre": NARRE,
    "mymodel": MyModel,
    "rgcl": RGCL,
    "mymodel_cfonly": MyModelCFOnly,
    "mymodel_neumf": MyModelNueMF,
    "mymodel_v2": MyModelV2,
    "letter": LetterModel,
}
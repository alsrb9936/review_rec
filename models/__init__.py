from models.base_model import BaseModel
from models.neumf import NeuMF
from models.deepconn import DeepCoNN
from models.narre import NARRE
from models.mymodel import MyModel
from models.mymodel_cfonly import MyModelCFOnly
from models.mymodel_concat import MyModelConcat
from models.mymodel_shared import MyModelShared
from models.mymodel_full import MyModelFull

MODEL_DICT = {
    "neumf": NeuMF,
    "deepconn": DeepCoNN,
    "narre": NARRE,
    "mymodel": MyModel,
    "mymodel_cfonly": MyModelCFOnly,
    "mymodel_concat": MyModelConcat,
    "mymodel_shared": MyModelShared,
    "mymodel_full": MyModelFull,
}
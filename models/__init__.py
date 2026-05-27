from models.base_model import BaseModel
from models.neumf import NeuMF
from models.deepconn import DeepCoNN

MODEL_DICT = {
    "neumf": NeuMF,
    "deepconn": DeepCoNN,
}
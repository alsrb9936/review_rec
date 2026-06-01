from dataset.base_dataset import BaseDataset
from dataset.neumf_dataset import NeuMFDataset
from dataset.deepconn_dataset import DeepCoNNDataset
from dataset.narre_dataset import NARREDataset
from dataset.mymodel_dataset import MyModelDataset
from dataset.rgcl_dataset import RGCLDataset
from dataset.letter_dataset import LetterDataset
DATASET_DICT = {
    "neumf": NeuMFDataset,
    "deepconn": DeepCoNNDataset,
    "narre": NARREDataset,
    "mymodel": MyModelDataset,
    "rgcl": RGCLDataset,
    "mymodel_cfonly": MyModelDataset,
    "mymodel_neumf": MyModelDataset,
    "mymodel_v2": NARREDataset,
    "letter": LetterDataset,
}
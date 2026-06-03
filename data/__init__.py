from data.neumf_dataset import NeuMFDataset
from data.deepconn_dataset import DeepCoNNDataset
from data.narre_dataset import NARREDataset
from data.rgcl_dataset import RGCLDataset
from data.daml_dataset import DAMLDataset
from data.lightgcn_dataset import LightGCNDataset
from data.transnet_dataset import TransNetDataset
from data.letter_dataset import LETTERDataset
from data.mymodel_v1_dataset import MyModelV1Dataset
from data.recafr_dataset import RecAFRDataset

DATASET_DICT = {
    "neumf": NeuMFDataset,
    "deepconn": DeepCoNNDataset,
    "narre": NARREDataset,
    "rgcl": RGCLDataset,
    "daml": DAMLDataset,
    "lightgcn": LightGCNDataset,
    "transnet": TransNetDataset,
    "mymodel_v1": MyModelV1Dataset,
    "letter": LETTERDataset,
    "recafr": RecAFRDataset,
}

from data.neumf_dataset import NeuMFDataset
from data.deepconn_dataset import DeepCoNNDataset
from data.narre_dataset import NARREDataset
from data.rgcl_dataset import RGCLDataset
from data.daml_dataset import DAMLDataset
from data.lightgcn_dataset import LightGCNDataset
from data.transnet_dataset import TransNetDataset

DATASET_DICT = {
    "neumf": NeuMFDataset,
    "deepconn": DeepCoNNDataset,
    "narre": NARREDataset,
    "rgcl": RGCLDataset,
    "daml": DAMLDataset,
    "lightgcn": LightGCNDataset,
    "transnet": TransNetDataset,
}

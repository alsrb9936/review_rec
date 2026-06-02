from dataset.base_dataset import BaseDataset
from review_rec.review_reproducibility.data.neumf_dataset import NeuMFDataset
from dataset.deepconn_dataset import DeepCoNNDataset
from dataset.narre_dataset import NARREDataset
from dataset.mymodel_dataset import MyModelDataset
from dataset.rgcl_dataset import RGCLDataset
from dataset.daml_dataset import DAMLDataset
from dataset.lightgcn_dataset import LightGCNDataset
DATASET_DICT = {
    "neumf": NeuMFDataset,
    "deepconn": DeepCoNNDataset,
    "narre": NARREDataset,
    "rgcl": RGCLDataset,
    "daml": DAMLDataset,
    "lightgcn": LightGCNDataset,
}

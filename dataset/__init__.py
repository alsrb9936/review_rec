from dataset.base_dataset import BaseDataset
from dataset.neumf_dataset import NeuMFDataset
from dataset.deepconn_dataset import DeepCoNNDataset

DATASET_DICT = {
    "neumf": NeuMFDataset,
    "deepconn": DeepCoNNDataset,
}
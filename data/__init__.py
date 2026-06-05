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
from data.rmg_dataset import RMGDataset
from data.mymodel_v2_dataset import MyModelV2Dataset
from data.mymodel_v3_dataset import MyModelV3Dataset
from data.mymodel_v4_dataset import MyModelV4Dataset
from data.mymodel_v5_dataset import MyModelV5Dataset
from data.cfarg_dataset import CFARGDataset
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
    "rmg": RMGDataset,
    "mymodel_v2": MyModelV2Dataset,
    "mymodel_v3": MyModelV3Dataset,
    "mymodel_v4": MyModelV4Dataset,
    "mymodel_v5": MyModelV5Dataset,
    "cfarg": CFARGDataset,
}

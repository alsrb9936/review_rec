
from trainer.base_trainer import BaseTrainer


class MyModelTrainer(BaseTrainer):
    def __init__(self, cfg, dataset, device):
        super().__init__(cfg, dataset, device)
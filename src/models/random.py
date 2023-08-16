import logging

import torch
from torch.utils.data import Dataset

from settings import OUTPUT_LAYER_NODES
from src.models.baseline import Baseline
from src.utils.commons import force_open, calculate_metrics_extended


logger = logging.getLogger()

# A model that generates random values as output
class RandomModel(Baseline, torch.nn.Module):

    def __init__(self, *args, **kwargs):
        Baseline.__init__(self, *args, **kwargs)
        torch.nn.Module.__init__(self)

    @classmethod
    def short_name(cls) -> str:
        return "basic-feedforward"

    def forward(self, x):
        raise torch.rand((x.shape[0], OUTPUT_LAYER_NODES))

    def learn(self, epoch_num: int, batch_size: int, splits: list, train_dataset: Dataset, weights_checkpoint_path: str=None, condition_save_threshold=0.9):
        return super().learn()
    
    def test(self, test_dataset, weights_checkpoint_path):
        return super().test()


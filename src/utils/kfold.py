import torch


class StratifiedKFold():

    def __init__(self, k_splits) -> None:
        self.k_splits = k_splits
    
    def split(self, dataset):
        for entry in dataset:
            
            pass

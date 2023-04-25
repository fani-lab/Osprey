from src.utils.commons import RegisterableObject

import torch


class BaseLossCalculator(RegisterableObject):
    pass

class WeightedBinaryCrossEntropy(BaseLossCalculator):

    def __init__(self, pos_weight=1, *args, **kwargs) -> None:
        super().__init__()
        self.pos_weight = pos_weight
    
    @classmethod
    def short_name(cls) -> str:
        return "weighted-binary-cross-entropy"

    def __call__(self, predictions, targets):
        _p = torch.clamp(predictions, 1e-7, 1 - 1e-7)
        result = (-targets * torch.log(_p) * self.pos_weight + (1 - targets) * - torch.log(1 - _p)).sum()
        return result

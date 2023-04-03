from src.utils.commons import RegisterableObject

import torch


class BaseLossCalculator(RegisterableObject):
    pass

class WeightedBinaryCrossEntropy(BaseLossCalculator):

    def __init__(self, pos_weight=1) -> None:
        super().__init__()
        self.pos_weight = pos_weight
    
    @classmethod
    def short_name(cls) -> str:
        return "weighted-binary-cross-entropy"

    def __call__(self, predictions, targets):
        result = (-targets * torch.log(predictions) * self.pos_weight + (1 - targets) * - torch.log(1 - predictions)).mean()
        return result

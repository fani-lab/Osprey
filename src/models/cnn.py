import torch

from src.models import ANNModule


class EbrahimiCNN(ANNModule):

    def __init__(self, input_size, activation, loss_func, lr, module_session_path, device="cpu"):

        super().__init__([], None, loss_func, lr, input_size, module_session_path, device)
        
        del self.model_stack

        self.cnn = torch.nn.Conv1d(input_size, 2000, 1, groups=1)
        self.out = torch.nn.Linear(2000, 2)
        self.activation = activation


    def forward(self, x):
        if x.is_sparse:
            x = x.to_dense()
        x = self.activation(self.cnn(x))
        x = torch.squeeze(x)
        x = self.out(x)
        x = torch.softmax(x, dim=1)

        return x
    
    @classmethod
    def short_name(cls) -> str:
        return "ebrahimi-cnn"


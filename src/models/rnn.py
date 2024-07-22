import logging
from glob import glob
import re
from typing import List

import torch.utils
import torch.utils.data

from src.models.baseline import PytorchBaseline
import settings

from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

from src.utils.commons import force_open, padding_collate_sequence_batch

logger = logging.getLogger()


class BaseRnnModule(PytorchBaseline):
    
    def __init__(self, hidden_size, num_layers, *args, **kwargs):
        PytorchBaseline.__init__(self, *args, **kwargs)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.snapshot_steps = 2
        self.core = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, nonlinearity='tanh',
                          batch_first=True)
        self.hidden2out = nn.Linear(in_features=self.hidden_size, out_features=settings.OUTPUT_LAYER_NODES)

    
    @classmethod
    def short_name(cls) -> str:
        return "base-rnn"

    def forward(self, x):
        out, _ = self.core(x)
        y_hat = self.hidden2out(out[:, -1])
        # y_hat = torch.sigmoid(y_hat)
        if y_hat.isnan().sum() > 0:
            print(end="")
        return y_hat

    def get_session_path(self, *args):
        return f"{self.session_path}" + self.__class__.short_name() + "/" + "/".join([str(a) for a in args])

    def get_detailed_session_path(self, dataset, *args):
        details = str(dataset) + "-" + str(self)
        return self.get_session_path(details, *args)
    
    def check_stop_early(self, *args, **kwargs):
        return kwargs.get("f2score", 0.0) >= 0.95 and self.early_stop
    
    def get_all_folds_checkpoints(self, dataset: Dataset):
        # main_path = glob(self.get_detailed_session_path(dataset, "weights", f"f{fold}", f"model_f{fold}.pth"))
        main_path = glob(self.get_detailed_session_path(dataset, "weights", "f[0-9]", "model_f[0-9].pth")) # Supports upto 10 folds (from 0 to 9)
        paths = [ pp for pp in main_path if re.search(r"model_f\d{1,2}.pth$", pp)]
        if len(paths) == 0:
            raise RuntimeError("no checkpoint was found. probably the model has not been trained.")
        return paths

    def get_new_optimizer(self, lr: float, *args, **kwargs):
        return torch.optim.Adam(self.parameters(), lr=lr)
    
    def get_new_scheduler(self, optimizer: torch.optim.Optimizer, *args, **kwargs):
        scheduler_args = {"verbose":False, "min_lr":1e-9, "threshold": 20, "cooldown": 5, "patience": 20, "factor":0.25, "mode": "min"}
        logger.debug(f"scheduler settings: {scheduler_args}")
        return ReduceLROnPlateau(optimizer, **scheduler_args)
    
    def get_dataloaders(self, dataset: Dataset, train_ids: List[int], validation_ids: List[int], batch_size: int):
        train_subsampler = SubsetRandomSampler(train_ids)
        validation_subsampler = SubsetRandomSampler(validation_ids)
        train_loader = DataLoader(dataset, batch_size=batch_size, drop_last=False,
                                                    sampler=train_subsampler, collate_fn=padding_collate_sequence_batch)
        validation_loader = DataLoader(dataset, batch_size=batch_size, drop_last=False,
                                                        sampler=validation_subsampler, collate_fn=padding_collate_sequence_batch)
        return train_loader, validation_loader

    def get_test_dataloaders(self, dataset: torch.utils.data.Dataset, batch_size: int) -> DataLoader:
        return DataLoader(dataset, batch_size=batch_size, collate_fn=padding_collate_sequence_batch)

    def save(self, path):
        with force_open(path, "wb") as f:
            torch.save(self.state_dict(), f)
            logger.info(f"saving sanpshot at {path}")

    def load_params(self, path):
        self.load_state_dict(torch.load(path))
        logger.info(f"loaded model weights from file: {path}")
    
    def __str__(self) -> str:
        return "lr"+ format(self.init_lr, "f") + "-h" + str(self.hidden_size) + "-l" + str(self.num_layers)


class LSTMModule(BaseRnnModule):

    @classmethod
    def short_name(cls) -> str:
        return "lstm"

    def __init__(self, hidden_size, num_layers, *args, **kwargs):
        super().__init__(hidden_size, num_layers, *args, **kwargs)
        self.core = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)


class GRUModule(BaseRnnModule):

    @classmethod
    def short_name(cls) -> str:
        return "gru"

    def __init__(self, hidden_size, num_layers, *args, **kwargs):
        super().__init__(hidden_size, num_layers, *args, **kwargs)
        self.core = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)


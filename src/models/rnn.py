import pickle
import logging
import shutil

from src.models.baseline import Baseline
from settings import settings

from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np

from src.utils.commons import force_open, calculate_metrics_extended, padding_collate_sequence_batch

logger = logging.getLogger()


class BaseRnnModule(Baseline, nn.Module):
    
    def __init__(self, hidden_size, num_layers, *args, **kwargs):
        nn.Module.__init__(self)
        Baseline.__init__(self, *args, **kwargs)

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
        out, hn = self.core(x)
        y_hat = self.hidden2out(out[:, -1])
        # y_hat = torch.sigmoid(y_hat)
        if y_hat.isnan().sum() > 0:
            print(end="")
        return hn, y_hat

    def get_session_path(self, *args):
        return f"{self.session_path}" + self.__class__.short_name() + "/" + "/".join([str(a) for a in args])

    def get_detailed_session_path(self, dataset, *args):
        details = str(dataset) + "-" + str(self)
        return self.get_session_path(details, *args)
    
    def reset_modules(self, module, parents_modules_names=[]):
        for name, module in module.named_children():
            if name in settings.IGNORED_PARAM_RESET:
                continue
            if isinstance(module, nn.ModuleList):
                self.reset_modules(module, parents_modules_names=[*parents_modules_names, name])
            elif isinstance(module, nn.Dropout):
                continue
            else:
                logger.info(f"resetting module parameters {'.'.join([name, *parents_modules_names])}")
                module.reset_parameters()

    def get_new_optimizer(self, lr, *args, **kwargs):
        return torch.optim.Adam(self.parameters(), lr=lr)
    
    def get_new_scheduler(self, optimizer, *args, **kwargs):
        scheduler_args = {"verbose":False, "min_lr":1e-9, "threshold": 20, "cooldown": 5, "patience": 20, "factor":0.25, "mode": "min"}
        logger.debug(f"scheduler settings: {scheduler_args}")
        return ReduceLROnPlateau(optimizer, **scheduler_args)

    def learn(self, epoch_num:int , batch_size: int, splits: list, train_dataset: Dataset, weights_checkpoint_path: str=None):
        if weights_checkpoint_path is not None and len(weights_checkpoint_path):
            self.load_params(weights_checkpoint_path)
        
        logger.info("training phase started")
        folds_metrics = []
        for fold, (train_ids, validation_ids) in enumerate(splits):
            self.optimizer = self.get_new_optimizer(self.init_lr)
            self.scheduler = self.get_new_scheduler(self.optimizer)
            logger.info(self.optimizer)
            logger.info(self.scheduler)
            logger.info(f'fetching data for fold #{fold}')
            train_subsampler = SubsetRandomSampler(train_ids)
            validation_subsampler = SubsetRandomSampler(validation_ids)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False,
                                                       sampler=train_subsampler, collate_fn=padding_collate_sequence_batch)
            validation_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False,
                                                            sampler=validation_subsampler, collate_fn=padding_collate_sequence_batch)
            last_lr = self.init_lr
            total_loss = []
            total_validation_loss = []
            # resetting module parameters
            self.reset_modules(module=self)
            
            # Train phase
            for i in range(1, epoch_num + 1):
                loss = 0
                epoch_loss = 0
                self.train()
                if self.optimizer.param_groups[0]["lr"] != last_lr:
                    logger.info(f"fold: {fold} | epoch: {i} | Learning rate changed from: {last_lr} -> {self.optimizer.param_groups[0]['lr']}")
                    last_lr = self.optimizer.param_groups[0]["lr"]
                for batch_index, (X, y) in enumerate(train_loader):
                    X = X.to(self.device)
                    y = y.to(self.device)
                    _, y_hat = self.forward(X)
                    y_hat = y_hat.reshape(-1)
                    self.optimizer.zero_grad()
                    loss = self.loss_function(y_hat, y)
                    if loss.isnan():
                        print(end="")
                    loss.backward()
                    epoch_loss += loss.item()
                    self.optimizer.step()
                    logger.debug(f"fold: {fold} | epoch: {i} | batch: {batch_index} | loss: {loss/X.shape[0]}")
                epoch_loss /= len(train_ids)
                total_loss.append(epoch_loss)
                # Validation phase
                all_preds = []
                all_targets = []
                validation_loss = 0
                self.eval()
                with torch.no_grad():
                    for batch_index, (X, y) in enumerate(validation_loader):
                        X = X.to(self.device)
                        y = y.to(self.device)
                        _, y_hat = self.forward(X)
                        y_hat = y_hat.reshape(-1)
                        loss = self.loss_function(y_hat, y)
                        validation_loss += loss.item()
                        all_preds.extend(torch.sigmoid(y_hat) if isinstance(self.loss_function, nn.BCEWithLogitsLoss) else y_hat)
                        all_targets.extend(y)
                    validation_loss /= len(validation_ids)
                    total_validation_loss.append(validation_loss)
                all_preds = torch.tensor(all_preds)
                all_targets = torch.tensor(all_targets)
                accuracy_value, recall_value, precision_value, f2score = calculate_metrics_extended(all_preds, all_targets, device=self.device)
                logger.info(f"fold: {fold} | epoch: {i} | train -> loss: {(epoch_loss):>0.5f} | validation -> loss: {(validation_loss):>0.5f} | accuracy: {(100 * accuracy_value):>0.6f} | precision: {(100 * precision_value):>0.6f} | recall: {(100 * recall_value):>0.6f} | f2: {(100 * f2score):>0.6f}")
                self.scheduler.step(validation_loss)
                self.train()
            folds_metrics.append((accuracy_value, precision_value, recall_value, f2score))
            snapshot_path = self.get_detailed_session_path(train_dataset, "weights", f"f{fold}", f"model_fold{fold}.pth")
            self.save(snapshot_path)
            plt.clf()
            plt.plot(np.arange(1, 1 + len(total_loss)), np.array(total_loss), "-r", label="training")
            plt.plot(np.arange(1, 1 + len(total_loss)), np.array(total_validation_loss), "-b", label="validation")
            plt.legend()
            plt.title(f"fold #{fold}")
            with force_open(self.get_detailed_session_path(train_dataset, "figures", f"loss_f{fold}.png"), "wb") as f:
                plt.savefig(f, dpi=300)
        MAHAK = 3
        max_metric = (0, folds_metrics[0][MAHAK])
        for i in range(1, len(folds_metrics)):
            if folds_metrics[i][MAHAK] > max_metric[1]:
                max_metric = (i, folds_metrics[i][MAHAK])
        logger.info(f"best model of cross validation for current training phase: fold #{max_metric[0]} with metric value of '{max_metric[1]}'")
        best_model_dest = self.get_detailed_session_path(train_dataset, "weights", f"best_model.pth")
        best_model_src = self.get_detailed_session_path(train_dataset, "weights", f"f{max_metric[0]}", f"model_fold{max_metric[0]}.pth")
        shutil.copyfile(best_model_src, best_model_dest)

    def test(self, test_dataset, weights_checkpoint_path):
        self.load_params(weights_checkpoint_path)
        all_preds = []
        all_targets = []
        test_dataloader = DataLoader(test_dataset, batch_size=64, collate_fn=padding_collate_sequence_batch)
        self.eval()
        with torch.no_grad():
            for X, y in test_dataloader:
                X = X.to(self.device)
                y = y.to(self.device)
                last_hidden, y_hat = self.forward(X)
                y_hat = y_hat.reshape(-1)
                all_preds.extend(torch.sigmoid(y_hat) if isinstance(self.loss_function, nn.BCEWithLogitsLoss) else y_hat)
                all_targets.extend(y)

        all_preds = torch.tensor(all_preds)
        all_targets = torch.tensor(all_targets)
        with force_open(self.get_detailed_session_path(test_dataset, 'preds.pkl'), 'wb') as file:
            pickle.dump(all_preds, file)
            logger.info(f'predictions are saved at: {file.name}')
        with force_open(self.get_detailed_session_path(test_dataset, 'targets.pkl'), 'wb') as file:
            pickle.dump(all_targets, file)
            logger.info(f'targets are saved at: {file.name}')

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


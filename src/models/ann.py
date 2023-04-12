import pickle
import logging

import torchmetrics
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.baseline import Baseline
from src.utils.commons import force_open
from settings import settings

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import KFold, StratifiedKFold

logger = logging.getLogger()


class ANNModule(Baseline, torch.nn.Module):

    def __init__(self, dimension_list, activation, loss_func, lr, input_size, module_session_path,
                 device='cpu', **kwargs):
        Baseline.__init__(self, input_size=input_size)
        torch.nn.Module.__init__(self)
        
        self.init_lr = lr
        self.dimension_list = dimension_list

        
        self.i2h = nn.Linear(input_size,
        dimension_list[0] if len(dimension_list) > 0 else 2)
        self.layers = nn.ModuleList()
        for i, j in zip(dimension_list, dimension_list[1:]):
            self.layers.append(nn.Linear(in_features=i, out_features=j))
        self.h2o = torch.nn.Linear(dimension_list[-1] if len(dimension_list) > 0 else input_size, 2)
        self.activation = activation

        self.loss_function = loss_func

        self.session_path = module_session_path if module_session_path[-1] == "\\" or module_session_path[
            -1] == "/" else module_session_path + "/"

        self.snapshot_steps = 2
        self.device = device

    @classmethod
    def short_name(cls) -> str:
        return "ann"

    def forward(self, x):
        """

        Args:
            x: Tensor object

        Returns: prediction of the model

        """
        x = self.activation(self.i2h(x))
        for layer in self.layers:
            x = self.activation(layer(x))

        x = self.h2o(x)
        x = torch.softmax(x, dim=1)
        x = torch.clamp(x, min=1e-12, max=1 - 1e-12)
        return x

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
            else:
                logger.info(f"resetting module parameters {'.'.join([name, *parents_modules_names])}")
                module.reset_parameters()

    def learn(self, epoch_num: int, batch_size: int, k_fold: int, train_dataset: Dataset):

        accuracy = torchmetrics.Accuracy('multiclass', num_classes=2, top_k=1).to(self.device)
        precision = torchmetrics.Precision('multiclass', num_classes=2, top_k=1).to(self.device)
        recall = torchmetrics.Recall('multiclass', num_classes=2, top_k=1).to(self.device)
        logger.info("training phase started")
        scheduler_args = {"verbose":True, "min_lr":1e-6, "threshold":8e-3, "patience":0, "factor":0.075}
        # kfold = KFold(n_splits=k_fold)
        kfold = StratifiedKFold(n_splits=k_fold, shuffle=True)
        xs, ys = [0] * len(train_dataset), [0] * len(train_dataset)
        for i in range(len(train_dataset)):
            entry = train_dataset[i]
            xs[i] = entry[0]
            ys[i] = entry[1]
        xs = torch.stack(xs)
        ys = torch.stack(ys)
        for fold, (train_ids, validation_ids) in enumerate(kfold.split(xs, ys.argmax(dim=1))):
            logger.info("Resetting Optimizer, Learning rate, and Scheduler")
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.init_lr, momentum=0.9)
            self.scheduler = ReduceLROnPlateau(self.optimizer, **scheduler_args)
            logger.debug(f"scheduler settings: {scheduler_args}")
            logger.info(f'fetching data for fold #{fold}')
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            validation_subsampler = torch.utils.data.SubsetRandomSampler(validation_ids)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                       sampler=train_subsampler)
            validation_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                            sampler=validation_subsampler)
            # Train phase
            total_loss = []
            # resetting module parameters
            self.reset_modules(module=self)
            for i in range(epoch_num):
                loss = 0
                for batch_index, (X, y) in enumerate(train_loader):
                    X = X.to(self.device)
                    y = y.to(self.device)
                    y_hat = self.forward(X)
                    self.optimizer.zero_grad()
                    loss = self.loss_function(y_hat, y)
                    loss.backward()
                    self.optimizer.step()
                    logger.info(f"fold: {fold} | epoch: {i} | batch: {batch_index} | loss: {loss}")
                    total_loss.append(loss.item())
                self.scheduler.step(loss)
            # Validation phase
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for batch_index, (X, y) in enumerate(validation_loader):
                    X = X.to(self.device)
                    y = y.to(self.device)
                    pred = self.forward(X)
                    all_preds.extend(pred)
                    all_targets.extend(y)
            all_preds = torch.stack(all_preds)
            all_targets = torch.stack(all_targets)
            logger.info(f'Validation Accuracy: {(100 * accuracy(all_preds, all_targets)):>0.1f}')
            logger.info(f'Validation precision: {(100 * precision(all_preds, all_targets)):>0.1f}')
            logger.info(f'Validation Recall: {(100 * recall(all_preds, all_targets)):>0.1f}')

            snapshot_path = self.get_detailed_session_path(train_dataset, "weights", f"f{fold}", f"model_fold{fold}.pth")
            self.save(snapshot_path)
            plt.clf()
            plt.plot(np.array(total_loss))
            # plt.axis([0, len(total_loss), 0, 1])
            with force_open(self.get_detailed_session_path(train_dataset, "figures", f"f{fold}", f"model_fold{fold}_loss.png"), "wb") as f:
                plt.savefig(f)
            # plt.show()

    def test(self, test_dataset):
        all_preds = []
        all_targets = []
        test_dataset.to(self.device)
        test_dataloader = DataLoader(test_dataset, batch_size=64)
        with torch.no_grad():
            for X, y in test_dataloader:
                pred = self.forward(X)
                all_preds.extend(pred)
                all_targets.extend(y)

        all_preds = torch.stack(all_preds)
        all_targets = torch.stack(all_targets)
        with force_open(self.get_detailed_session_path(test_dataset, 'preds.pkl'), 'wb') as file:
            pickle.dump(all_preds, file)
            logger.info(f'predictions are saved at {file.name}.')
        with force_open(self.get_detailed_session_path(test_dataset, 'targets.pkl'), 'wb') as file:
            pickle.dump(all_targets, file)
            logger.info(f'targets are saved at {file.name}.')

    def save(self, path):
        with force_open(path, "wb") as f:
            torch.save(self.state_dict(), f)
            logger.info(f"saving model at {path}")

    def load_params(self, path):
        try:
            self.load_state_dict(torch.load(path))
            logger.info("parameters loaded successfully")
        except Exception as e:
            logger.debug(e)

    def __str__(self) -> str:
        return str(self.init_lr) + "-" + ".".join((str(l) for l in self.dimension_list))
    
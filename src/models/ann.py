import pickle
import logging
import time

import torchmetrics
from models.baseline import Baseline
from preprocessing.base import BasePreprocessing
from utils.commons import force_open

import torch
from torch import nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import KFold

logger = logging.getLogger()


class ANNModule(torch.nn.Module, Baseline):

    def __init__(self, dimension_list, activation, loss_func, lr, train_dataset, module_session_path,
                 number_of_classes=2, **kwargs):

        super(ANNModule, self).__init__()
        self.number_of_classes = number_of_classes
        self.i2h = nn.Linear(train_dataset.shape[1],
                             dimension_list[0] if len(dimension_list) > 0 else self.number_of_classes)
        self.layers = nn.ModuleList()
        for i, j in zip(dimension_list, dimension_list[1:]):
            self.layers.append(nn.Linear(in_features=i, out_features=j))
        self.h2o = torch.nn.Linear(dimension_list[-1] if len(dimension_list) > 0 else train_dataset.shape[1],
                                   self.number_of_classes)
        self.activation = activation
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        self.loss_function = loss_func

        self.train_dataset = train_dataset
        self.session_path = module_session_path if module_session_path[-1] == "\\" or module_session_path[
            -1] == "/" else module_session_path + "/"

        self.snapshot_steps = 2

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
        return x

    def get_session_path(self, *args):
        return f"{self.session_path}" + "ann/" + "/".join([str(a) for a in args])

    def learn(self, epoch_num: int, batch_size: int, k_fold: int):
        accuracy = torchmetrics.Accuracy('binary', )
        precision = torchmetrics.Precision('binary', )
        recall = torchmetrics.Recall('binary', )
        logger.info("training phase started")
        kfold = KFold(n_splits=k_fold)
        for fold, (train_ids, validation_ids) in enumerate(kfold.split(self.train_dataset)):
            logger.info(f'FOLD {fold}')
            logger.info('--------------------------------')
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            validation_subsampler = torch.utils.data.SubsetRandomSampler(validation_ids)
            train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size,
                                                       sampler=train_subsampler)
            validation_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size,
                                                            sampler=validation_subsampler)
            # Train phase
            total_loss = []
            # resetting module parameters
            for name, module in self.named_children():
                try:
                    if isinstance(module, nn.ModuleList):
                        for name_, layer in module.named_children():
                            print(name_)
                            layer.reset_parameters()
                            print("parameters reset!")
                    else:
                        print(name)
                        module.reset_parameters()
                        print("parameters reset!")
                except:
                    pass
            for i in range(epoch_num):
                loss = 0
                for batch_index, (X, y) in enumerate(train_loader):
                    y_hat = self.forward(X)
                    self.optimizer.zero_grad()
                    loss = self.loss_function(y_hat, y)
                    loss.backward()
                    self.optimizer.step()
                    logger.info(f"epoch: {i} | batch: {batch_index} | loss: {loss}")
                total_loss.append(loss.item())
                logger.info(f'epoch {i}:\n Loss: {loss}\n')
                # Validation phase
                all_preds = []
                all_targets = []
                size = len(validation_loader)
                num_batches = len(validation_loader)
                test_loss, correct = 0, 0
                with torch.no_grad():
                    for batch_index, (X, y) in enumerate(validation_loader):
                        pred = self.forward(X)
                        all_preds.extend(pred.argmax(1))
                        all_targets.extend(y)
                        test_loss += self.loss_function(pred, y).item()
                        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                test_loss /= num_batches
                correct /= size
                all_preds = torch.tensor(all_preds)
                all_targets = torch.tensor(all_targets)
                logger.info(f"Validation Error: Avg loss: {test_loss:>8f}")
                logger.info(f'torchmetrics Accuracy: {(100 * accuracy(all_preds, all_targets)):>0.1f}')
                logger.info(f'torchmetrics precision: {(100 * precision(all_preds, all_targets)):>0.1f}')
                logger.info(f'torchmetrics Recall: {(100 * recall(all_preds, all_targets)):>0.1f}')

            snapshot_path = self.get_session_path(f"f{fold}", f"model_fold{fold}.pth")
            self.save(snapshot_path)
            plt.plot(np.array(total_loss))
            plt.axis([0, epoch_num, 0, 1])
            plt.savefig(self.get_session_path(f"f{fold}", f"model_fold{fold}_loss.png"))
            plt.show()

    def test(self, test_dataset):
        accuracy = torchmetrics.Accuracy('binary', )
        precision = torchmetrics.Precision('binary', )
        recall = torchmetrics.Recall('binary', )
        all_preds = []
        all_targets = []
        test_dataloader = DataLoader(test_dataset, batch_size=64)
        size = len(test_dataset)
        num_batches = len(test_dataloader)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in test_dataloader:
                pred = self.forward(X)
                all_preds.extend(pred.argmax(1))
                all_targets.extend(y)
                test_loss += self.loss_function(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        all_preds = torch.tensor(all_preds)
        all_targets = torch.tensor(all_targets)
        logger.info(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        logger.info(f'torchmetrics Accuracy: {(100 * accuracy(all_preds, all_targets)):>0.1f}')
        logger.info(f'torchmetrics precision: {(100 * precision(all_preds, all_targets)):>0.1f}')
        logger.info(f'torchmetrics Recall: {(100 * recall(all_preds, all_targets)):>0.1f}')

    def save(self, path):
        with force_open(path, "wb") as f:
            torch.save(self, f)
            logger.info(f"saving model at {path}")

    def load_params(self, path):
        try:
            self.load_state_dict(torch.load(path))
            logger.info("parameters loaded successfully")
        except Exception as e:
            logger.debug(e)

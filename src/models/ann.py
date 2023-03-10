import pickle
import logging
import time

import torchmetrics
from models.baseline import Baseline
from preprocessing.base import BasePreprocessing

import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.model_selection import KFold

logger = logging.getLogger()


class SimpleANN(torch.nn.Module, Baseline):

    def __init__(self, dimension_list, activation, loss_func, lr, train_dataset, number_of_classes=2, **kwargs):

        super(SimpleANN, self).__init__()
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
            for i in range(1, epoch_num + 1):
                loss = 0
                for batch_index, (X, y) in enumerate(train_loader):
                    y_hat = self.forward(X)
                    self.optimizer.zero_grad()
                    loss = self.loss_function(y_hat, y)
                    loss.backward()
                    self.optimizer.step()
                    logger.info(f"epoch: {i} | batch: {batch_index} | loss: {loss}")

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
                logger.info(f'torchmetrics Recall: {(100 * recall(all_preds, all_targets)):>0.1f}\n')

        current_time = time.strftime("%m-%d-%Y-%H-%M", time.localtime())
        self.save(path=f"output/ann/ann-{current_time}.pth")

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

    def get_session_path(self, file_name: str = "") -> str:
        return self.preprocessed_path + file_name

    def save(self, path):
        try:
            torch.save(self.state_dict(), path)
            logger.info('parameters saved successfully.')
        except Exception as e:
            logger.debug(e)

    def load_params(self, path):
        try:
            self.load_state_dict(torch.load(path))
            logger.info("parameters loaded successfully")
        except Exception as e:
            logger.debug(e)

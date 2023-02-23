import pickle
import logging

import torchmetrics
from models.baseline import Baseline
from preprocessing.base import BasePreprocessing

import torch
from torch import nn
from torch.utils.data import DataLoader


logger = logging.getLogger()

class SimpleANN(torch.nn.Module, Baseline):

    def __init__(self, dimension_list, activation, loss_func, lr, train_dataset,**kwargs):

        super(SimpleANN, self).__init__()

        self.layers = nn.ModuleList()
        for i, j in zip(dimension_list, dimension_list[1:]):
            self.layers.append(nn.Linear(in_features=i, out_features=j))

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
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)

        x = self.layers[-1](x)
        return x

    def learn(self, epoch_num: int, batch_size: int):
        """
        This function is the training phase of the model

        Args:
             epoch_num: number of epochs
             batch_size: size of training batches

        """
        recall = torchmetrics.Recall(task='multiclass', num_classes=2)
        logger.info("training phase started")
        for i in range(1, epoch_num+1):
            loss = 0
            train_dataloader = DataLoader(self.train_dataset, batch_size, shuffle=True)
            for X, y in train_dataloader:
                y_hat = self.forward(X)
                self.optimizer.zero_grad()
                loss = self.loss_function(y_hat, y)
                loss.backward()
                self.optimizer.step()
                logger.info(f'recall on batch: {recall(y_hat.argmax(1), y)}')

            logger.info(f'epoch {i}:\n Loss: {loss}')

    def test(self):
        pass

    def get_session_path(self, file_name: str = "") -> str:
        return self.preprocessed_path + file_name

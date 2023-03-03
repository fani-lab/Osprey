import pickle
import logging
import time
import pandas as pd
import torchmetrics

from src.models.baseline import Baseline
from src.preprocessing.base import BasePreprocessing

import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger()


class RnnModule(nn.Module, Baseline):
    def __init__(self, input_size, hidden_dim, num_layers, activation, loss_func, lr, train_dataset,
                 learning_batch_size=1, number_of_classes=2, **kwargs):
        super(RnnModule, self).__init__()

        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, nonlinearity='relu',
                          batch_first=True)
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.hidden2out = nn.Linear(in_features=hidden_dim, out_features=number_of_classes)
        self.batch_size = learning_batch_size
        self.activation = activation
        self.loss_function = loss_func
        self.train_dataset = train_dataset
        self.number_of_classes = number_of_classes
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)

    def forward(self, x):
        h0 = torch.zeros(1 * self.num_layers, self.batch_size, self.hidden_size)
        out, hn = self.rnn(x, h0)
        out = self.hidden2out(out)
        # out = torch.softmax(out, dim=1)
        return out

    def learn(self, epoch_num=10, batch_size=64):
        logger.info("training phase started")
        for i in range(1, epoch_num + 1):
            loss = 0
            train_dataloader = DataLoader(self.train_dataset, batch_size, drop_last=True, shuffle=True)
            for batch_index, (X, y) in enumerate(train_dataloader):
                # search for sparse with rnn
                X = X.to_dense()
                X = X.unsqueeze(1)
                y_hat = self.forward(X)
                y_hat = y_hat.squeeze()
                self.optimizer.zero_grad()
                loss = self.loss_function(y_hat, y)
                loss.backward()
                self.optimizer.step()
                logger.info(f"epoch: {i} | batch: {batch_index} | loss: {loss}")

            logger.info(f'epoch {i}:\n Loss: {loss}')
        current_time = time.strftime("%m-%d-%Y-%H-%M", time.localtime())
        # self.save(path=f"output/ann/ann-{current_time}.pth")

    def test(self, test_dataset):
        accuracy = torchmetrics.Accuracy('binary', )
        precision = torchmetrics.Precision('binary', )
        recall = torchmetrics.Recall('binary', )
        all_preds = []
        all_targets = []
        test_dataloader = DataLoader(test_dataset, batch_size=64, drop_last=True)
        size = len(test_dataset)
        num_batches = len(test_dataloader)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in test_dataloader:
                X = X.to_dense()
                X = X.unsqueeze(1)
                pred = self.forward(X)
                pred = pred.squeeze()
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


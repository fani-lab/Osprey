import pickle
import logging
import time
import pandas as pd
import torchmetrics

from src.models.baseline import Baseline

import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from src.utils.commons import force_open

logger = logging.getLogger()


class RnnModule(Baseline, nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, activation, loss_func, lr, train_dataset,
                 learning_batch_size, module_session_path, number_of_classes=2, **kwargs):
        Baseline.__init__(self, input_size=input_size)
        nn.Module.__init__(self)

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
        self.session_path = module_session_path if module_session_path[-1] == "\\" or module_session_path[
            -1] == "/" else module_session_path + "/"
        self.snapshot_steps = 2
    
    @classmethod
    def short_name(cls) -> str:
        return "rnn"

    def forward(self, x):
        # h0 = torch.zeros(1 * self.num_layers, self.batch_size, self.hidden_size)
        out, hn = self.rnn(x)
        out = self.hidden2out(out)
        # out = torch.softmax(out, dim=1)
        out = torch.sigmoid(out)
        return out

    def get_session_path(self, *args):
        return f"{self.session_path}" + "rnn/" + "/".join([str(a) for a in args])

    def learn(self, epoch_num=10, batch_size=64, k_fold: int = 5):
        accuracy = torchmetrics.Accuracy('binary', )
        precision = torchmetrics.Precision('binary', )
        recall = torchmetrics.Recall('binary', )
        logger.info("training phase started")
        kfold = KFold(n_splits=k_fold)
        for fold, (train_ids, validation_ids) in enumerate(kfold.split(self.train_dataset)):
            logger.info(f'FOLD {fold + 1}')
            logger.info('--------------------------------')
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            validation_subsampler = torch.utils.data.SubsetRandomSampler(validation_ids)
            train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, drop_last=True,
                                                       sampler=train_subsampler)
            validation_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, drop_last=True,
                                                            sampler=validation_subsampler)
            total_loss = []
            valid_loss = []
            # resetting module parameters
            for name, module in self.named_children():
                try:
                    if isinstance(module, nn.ModuleList):
                        for name_, layer in module.named_children():
                            layer.reset_parameters()
                            logger.info("parameters reset")
                    else:
                        module.reset_parameters()
                        logger.info("parameters reset")
                except Exception as e:
                    logger.error(e)
            # Train phase
            for i in range(1, epoch_num + 1):
                loss = 0

                # train_dataloader = DataLoader(self.train_dataset, batch_size, drop_last=True, shuffle=True)
                for batch_index, (X, y) in enumerate(train_loader):
                    # search for sparse with rnn
                    y = y.type(torch.float)
                    X = X.to_dense()
                    X = X.unsqueeze(1)
                    y_hat = self.forward(X)
                    y_hat = y_hat.squeeze()
                    self.optimizer.zero_grad()
                    loss = self.loss_function(y_hat, y)
                    loss.backward()
                    self.optimizer.step()
                    logger.info(f"epoch: {i} | batch: {batch_index} | loss: {loss}")
                total_loss.append(loss.item())
                logger.info(f'epoch {i}:\n Loss: {loss}')
            # Validation phase
            all_preds = []
            all_targets = []
            size = len(validation_loader)
            num_batches = len(validation_loader)
            # test_loss, correct = 0, 0
            with torch.no_grad():
                for batch_index, (X, y) in enumerate(validation_loader):
                    y = y.type(torch.float)
                    X = X.to_dense()
                    X = X.unsqueeze(1)
                    pred = self.forward(X)
                    pred = pred.squeeze()
                    all_preds.extend(pred)
                    # all_preds.extend(pred.argmax(1))
                    all_targets.extend(y)
                    valid_loss.append(self.loss_function(pred, y).item())
                    # test_loss += self.loss_function(pred, y).item()
                    # correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            # test_loss /= num_batches
            # correct /= size
            all_preds = torch.tensor(all_preds)
            all_targets = torch.tensor(all_targets)
            # logger.info(f"Validation Error: Avg loss: {test_loss:>8f}")
            logger.info(f'torchmetrics Accuracy: {(100 * accuracy(all_preds, all_targets)):>0.1f}')
            logger.info(f'torchmetrics precision: {(100 * precision(all_preds, all_targets)):>0.1f}')
            logger.info(f'torchmetrics Recall: {(100 * recall(all_preds, all_targets)):>0.1f}')

            # saving the whole model at the end of each fold
            snapshot_path = self.get_session_path(f"f{fold}", f"model_fold{fold}.pth")
            self.save(snapshot_path)
            plt.plot(np.array(total_loss))
            # plt.axis([0, epoch_num, 0, 1])
            plt.title(f"fold{fold}_train_loss")
            plt.savefig(self.get_session_path(f"f{fold}", f"model_fold{fold}_loss.png"))
            plt.show()

    def test(self, test_dataset):
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
                # all_preds.extend(pred.argmax(1))
                all_preds.extend(pred)
                all_targets.extend(y)
                # test_loss += self.loss_function(pred, y).item()
                # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        # test_loss /= num_batches
        # correct /= size
        all_preds = torch.tensor(all_preds)
        all_targets = torch.tensor(all_targets)
        with force_open(self.get_session_path('preds.pkl'), 'wb') as file:
            pickle.dump(all_preds, file)
            logger.info('predictions are saved.')
        with force_open(self.get_session_path('targets.pkl'), 'wb') as file:
            pickle.dump(all_targets, file)
            logger.info('targets are saved.')

    def eval(self):
        Baseline.eval(self)

    def save(self, path):
        with force_open(path, "wb") as f:
            torch.save(self.state_dict(), f)
            logger.info(f"saving sanpshot at {path}")

    def load_params(self, path):
        try:
            self.load_state_dict(torch.load(path))
            logger.info("parameters loaded successfully")
        except Exception as e:
            logger.debug(e)

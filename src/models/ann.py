import pickle
import logging
import shutil

from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.baseline import Baseline
from src.utils.commons import force_open, calculate_metrics
from settings import settings

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import KFold, StratifiedKFold

logger = logging.getLogger()


class ANNModule(Baseline, torch.nn.Module):

    def __init__(self, dimension_list, dropout_list, activation, loss_func, lr, input_size, module_session_path, validation_steps=-1,
                 device='cpu', **kwargs):
        Baseline.__init__(self, input_size=input_size)
        torch.nn.Module.__init__(self)
        if len(dropout_list) > len(dimension_list):
            raise ValueError(f"the length of dropout_list should be less equal than that of dimension_list: {len(dropout_list)} > {len(dimension_list)} ")
        self.dropout_list = [0] * len(dimension_list)
        self.dropout_list[:len(dropout_list)] = dropout_list
        
        self.init_lr = lr
        self.dimension_list = dimension_list + [2]
        self.validation_steps = validation_steps
        
        self.i2h = nn.Linear(input_size,
        self.dimension_list[0] if len(self.dimension_list) > 0 else 2)
        torch.nn.init.normal_(self.i2h.weight)
        self.layers = nn.ModuleList()
        for i, j, d in zip(self.dimension_list, self.dimension_list[1:], self.dropout_list):
            l = nn.Linear(in_features=i, out_features=j)
            self.layers.append(nn.Dropout(d))
            torch.nn.init.normal_(l.weight)
            self.layers.append(l)
        # self.h2o = torch.nn.Linear(self.dimension_list[-1] if len(self.dimension_list) > 0 else input_size, 2)
        # torch.nn.init.normal_(self.h2o.weight)
        self.activation = activation

        self.loss_function = loss_func

        self.session_path = module_session_path if module_session_path[-1] == "\\" or module_session_path[
            -1] == "/" else module_session_path + "/"

        self.snapshot_steps = 2
        self.device = device

        logger.info(f"dimension list of nodes: {self.dimension_list}")
        logger.info(f"dropout list: {self.dropout_list}")

    @classmethod
    def short_name(cls) -> str:
        return "ann"

    def forward(self, x):
        """

        Args:
            x: Tensor object

        Returns: prediction of the model

        """
        x = self.i2h(x)
        for layer in self.layers:
            x = layer(self.activation(x))

        # x = self.h2o(x)
        x = torch.softmax(x, dim=1)
        # x = torch.clamp(x, min=1e-12, max=1 - 1e-12)
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
            elif isinstance(module, nn.Dropout):
                continue
            else:
                logger.info(f"resetting module parameters {'.'.join([name, *parents_modules_names])}")
                module.reset_parameters()

    def learn(self, epoch_num: int, batch_size: int, k_fold: int, train_dataset: Dataset):

        logger.info("training phase started")
        scheduler_args = {"verbose":False, "min_lr":1e-10, "threshold":1e-4, "patience":10, "factor":0.25}
        # kfold = KFold(n_splits=k_fold)
        kfold = StratifiedKFold(n_splits=k_fold, shuffle=True)
        xs, ys = [0] * len(train_dataset), [0] * len(train_dataset)
        for i in range(len(train_dataset)):
            entry = train_dataset[i]
            xs[i] = entry[0]
            ys[i] = entry[1]
        xs = torch.stack(xs)
        ys = torch.stack(ys)
        folds_metrics = []
        last_lr = self.init_lr
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
            validation_loader = torch.utils.data.DataLoader(train_dataset, batch_size=(256 if len(validation_ids) > 1024 else len(validation_ids)),
                                                            sampler=validation_subsampler)
            # Train phase
            total_loss = []
            # resetting module parameters
            self.reset_modules(module=self)
            for i in range(epoch_num):
                loss = 0
                epoch_loss = 0
                for batch_index, (X, y) in enumerate(train_loader):
                    self.optimizer.zero_grad()
                    X = X.to(self.device)
                    y = y.to(self.device)
                    y_hat = self.forward(X)
                    loss = self.loss_function(y_hat, y)
                    loss.backward()
                    self.optimizer.step()
                    logger.debug(f"fold: {fold} | epoch: {i} | batch: {batch_index} | loss: {loss}")
                    total_loss.append(loss.item())
                    epoch_loss += loss
                self.scheduler.step(loss)
                if self.optimizer.param_groups[0]["lr"] != last_lr:
                    logger.info(f"fold: {fold} | epoch: {i} | Learning rate changed from: {last_lr} -> {self.optimizer.param_groups[0]['lr']}")
                    last_lr = self.optimizer.param_groups[0]["lr"]
                
                if (self.validation_steps == -1 and epoch_num-1 == i) or i % self.validation_steps == 0 or epoch_num-1 == i:
                    # Validation phase
                    all_preds = []
                    all_targets = []
                    validation_loss = 0
                    with torch.no_grad():
                        for batch_index, (X, y) in enumerate(validation_loader):
                            X = X.to(self.device)
                            y = y.to(self.device)
                            pred = self.forward(X)
                            loss = self.loss_function(pred, y)
                            validation_loss += loss
                            all_preds.extend(pred)
                            all_targets.extend(y)
                    all_preds = torch.stack(all_preds)
                    all_targets = torch.stack(all_targets)
                    
                    accuracy_value, recall_value, precision_value = calculate_metrics(all_preds, all_targets)

                    logger.info(f"fold: {fold} | epoch: {i} | train -> loss: {(epoch_loss):>0.5f} | validation -> loss: {(validation_loss):>0.5f} | accuracy: {(100 * accuracy_value):>0.6f} | precision: {(100 * precision_value):>0.6f} | recall: {(100 * recall_value):>0.6f}")

            folds_metrics.append((accuracy_value, precision_value, recall_value))

            snapshot_path = self.get_detailed_session_path(train_dataset, "weights", f"f{fold}", f"model_fold{fold}.pth")
            self.save(snapshot_path)
            plt.clf()
            plt.plot(np.array(total_loss))
            # plt.axis([0, len(total_loss), 0, 1])
            with force_open(self.get_detailed_session_path(train_dataset, "figures", f"f{fold}", f"model_fold{fold}_loss.png"), "wb") as f:
                plt.savefig(f)
            # plt.show()
        max_metric = (0, folds_metrics[0][1])
        for i in range(1, len(folds_metrics)):
            if folds_metrics[i][1] > max_metric[1]:
                max_metric = (i, folds_metrics[i][1])
        logger.info(f"best model of cross validation for current training phase: fold #{max_metric[0]} with metric value of '{max_metric[1]}'")
        best_model_dest = self.get_detailed_session_path(train_dataset, "weights", f"best_model.pth")
        best_model_src = self.get_detailed_session_path(train_dataset, "weights", f"f{max_metric[0]}", f"model_fold{max_metric[0]}.pth")
        shutil.copyfile(best_model_src, best_model_dest)
        return best_model_dest

    def test(self, test_dataset, weights_checkpoint_path):
        checkpoint = torch.load(weights_checkpoint_path)
        self.load_state_dict(checkpoint.get("model", checkpoint))

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
        return str(self.init_lr) + "-" + ".".join((str(l) for l in self.dimension_list)) + "-" + ".".join((str(d) for d in self.dropout_list))
    
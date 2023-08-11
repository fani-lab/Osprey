import pickle
import logging
import re
from glob import glob

from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.baseline import Baseline
from src.utils.commons import force_open, calculate_metrics_extended
from src.utils.loss_functions import DynamicSuperLoss
import settings
from settings import OUTPUT_LAYER_NODES

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from transformers.modeling_utils import PreTrainedModel

import matplotlib.pyplot as plt
import numpy as np


logger = logging.getLogger()


class AbstractFeedForward(Baseline, torch.nn.Module):
    
    def __init__(self, *args, **kwargs):
        Baseline.__init__(self, *args, **kwargs)
        torch.nn.Module.__init__(self)

    @classmethod
    def short_name(cls) -> str:
        return "basic-feedforward"

    def forward(self, x):
        raise NotImplementedError()
    
    def get_session_path(self, *args):
        return f"{self.session_path}" + self.__class__.short_name() + "/" + "/".join([str(a) for a in args])
    
    def get_detailed_session_path(self, dataset, *args):
        details = str(dataset) + "-" + str(self)
        return self.get_session_path(details, *args)

    def get_dataloaders(self, dataset, train_ids, validation_ids, batch_size):
        train_subsampler = SubsetRandomSampler(train_ids)
        validation_subsampler = SubsetRandomSampler(validation_ids)
        train_loader = DataLoader(dataset, batch_size=batch_size,
                                                    sampler=train_subsampler)
        validation_loader = DataLoader(dataset, batch_size=(256 if len(validation_ids) > 1024 else len(validation_ids)),
                                       sampler=validation_subsampler)
        
        return train_loader, validation_loader

    def get_all_folds_checkpoints(self, dataset):
        main_path = glob(self.get_detailed_session_path(dataset, "weights", "f*", "model_f[0-9]+.pth"))
        paths = [ pp for pp in main_path if re.search(r"model_f\d{1,2}.pth$", pp)]
        if len(paths) == 0:
            raise RuntimeError("no checkpoint was found. probably the model has not been trained.")
        return paths

    def get_new_optimizer(self, lr, *args, **kwargs):
        # return torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        return torch.optim.Adam(self.parameters(), lr=lr)
    
    def get_new_scheduler(self, optimizer, *args, **kwargs):
        # scheduler_args = {"verbose":False, "min_lr":0, "threshold":1e-4, "patience":10, "factor":0.25}
        scheduler_args = {"verbose":False, "min_lr":1e-9, "threshold": 20, "cooldown": 5, "patience": 20, "factor":0.25, "mode": "min"}
        logger.debug(f"scheduler settings: {scheduler_args}")
        return ReduceLROnPlateau(optimizer, **scheduler_args)

    def reset_modules(self, module, parents_modules_names=[]):
        for name, module in module.named_children():
            if name in settings.ALL_IGNORED_PARAM_RESET:
                continue
            if isinstance(module, nn.ModuleList):
                self.reset_modules(module, parents_modules_names=[*parents_modules_names, name])
            elif isinstance(module, nn.Dropout) or isinstance(module, PreTrainedModel):
                continue
            else:
                logger.info(f"resetting module parameters {'.'.join([name, *parents_modules_names])}")
                module.reset_parameters()

    def learn(self, epoch_num: int, batch_size: int, splits: list, train_dataset: Dataset, weights_checkpoint_path: str=None, condition_save_threshold=0.9):
        if weights_checkpoint_path is not None and len(weights_checkpoint_path):
            checkpoint = torch.load(weights_checkpoint_path)
            self.load_state_dict(checkpoint.get("model", checkpoint))

        logger.info(f"saving epoch condition: f2score>{condition_save_threshold}")
        logger.info("training phase started")
        
        folds_metrics = []
        logger.info(f"number of folds: {len(splits)}")
        for fold, (train_ids, validation_ids) in enumerate(splits):
            self.train()
            logger.info("Resetting Optimizer, Learning rate, and Scheduler")
            self.optimizer = self.get_new_optimizer(self.init_lr)
            self.scheduler = self.get_new_scheduler(self.optimizer)
            last_lr = self.init_lr
            logger.info(f'fetching data for fold #{fold}')
            train_loader, validation_loader = self.get_dataloaders(train_dataset, train_ids, validation_ids, batch_size)
            # Train phase
            total_loss = []
            total_validation_loss = []
            # resetting module parameters
            self.reset_modules(module=self)
            for i in range(epoch_num):
                self.train()
                loss = 0
                epoch_loss = 0
                for (X, y) in (train_loader):
                    self.optimizer.zero_grad()
                    if isinstance(X, tuple) or isinstance(X, list):
                        X = [l.to(self.device) for l in X]
                    else:
                        X = X.to(self.device)
                    y = y.reshape(-1, 1).to(self.device)
                    y_hat = self.forward(X)
                    loss = self.loss_function(y_hat, y)
                    loss.backward()
                    self.optimizer.step()
                    # logger.debug(f"fold: {fold} | epoch: {i} | batch: {batch_index} | loss: {loss}")
                    epoch_loss += loss.item()
                epoch_loss /= len(train_ids)
                total_loss.append(epoch_loss)
                self.scheduler.step(loss)
                if self.optimizer.param_groups[0]["lr"] != last_lr:
                    logger.info(f"fold: {fold} | epoch: {i} | Learning rate changed from: {last_lr} -> {self.optimizer.param_groups[0]['lr']}")
                    last_lr = self.optimizer.param_groups[0]["lr"]

                all_preds = []
                all_targets = []
                validation_loss = 0
                self.eval()
                with torch.no_grad():
                    for batch_index, (X, y) in enumerate(validation_loader):
                        if isinstance(X, tuple) or isinstance(X, list):
                            X = [l.to(self.device) for l in X]
                        else:
                            X = X.to(self.device)
                        y = y.reshape(-1, 1).to(self.device)
                        pred = self.forward(X)
                        loss = self.loss_function(pred, y)
                        validation_loss += loss.item()
                        all_preds.extend(torch.sigmoid(pred) if isinstance(self.loss_function, nn.BCEWithLogitsLoss) else pred)
                        all_targets.extend(y)
                    validation_loss /= len(validation_ids)
                    total_validation_loss.append(validation_loss)
                all_preds = torch.stack(all_preds)
                all_targets = torch.stack(all_targets)
                
                accuracy_value, recall_value, precision_value, f2score, f05score = calculate_metrics_extended(all_preds, all_targets, device=self.device)
                logger.info(f"fold: {fold} | epoch: {i} | train -> loss: {(epoch_loss):>0.5f} | validation -> loss: {(validation_loss):>0.5f} | accuracy: {(100 * accuracy_value):>0.6f} | precision: {(100 * precision_value):>0.6f} | recall: {(100 * recall_value):>0.6f} | f2: {(100 * f2score):>0.6f} | f0.5: {(100 * f05score):>0.6f}")
                
                epoch_snapshot_path = self.get_detailed_session_path(train_dataset, "weights", f"f{fold}", f"model_f{fold}_e{i}.pth")
                if f2score >= condition_save_threshold:
                    logger.info(f"fold: {fold} | epoch: {i} | saving model at {epoch_snapshot_path}")
                    self.save(epoch_snapshot_path)

            folds_metrics.append((accuracy_value, precision_value, recall_value))
            
            snapshot_path = self.get_detailed_session_path(train_dataset, "weights", f"f{fold}", f"model_f{fold}.pth")
            self.save(snapshot_path)
            plt.clf()
            plt.plot(np.arange(1, 1 + len(total_loss)), np.array(total_loss), "-r", label="training")
            plt.plot(np.arange(1, 1 + len(total_loss)), np.array(total_validation_loss), "-b", label="validation")
            plt.legend()
            plt.title(f"fold #{fold}")
            with force_open(self.get_detailed_session_path(train_dataset, "figures", f"loss_f{fold}.png"), "wb") as f:
                plt.savefig(f, dpi=300)
    
    def test(self, test_dataset, weights_checkpoint_path):
        for i, path in enumerate(weights_checkpoint_path):
            logger.info(f"testing checkpoint at: {path}")
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint.get("model", checkpoint))

            all_preds = []
            all_targets = []
            test_dataset.to(self.device)
            test_dataloader = DataLoader(test_dataset, batch_size=64)
            self.eval()
            with torch.no_grad():
                for X, y in test_dataloader:
                    if isinstance(X, tuple) or isinstance(X, list):
                        X = [l.to(self.device) for l in X]
                    else:
                        X = X.to(self.device)
                    y = y.reshape(-1, 1).to(self.device)
                    pred = self.forward(X)
                    all_preds.extend(torch.sigmoid(pred) if isinstance(self.loss_function, nn.BCEWithLogitsLoss) else pred)
                    all_targets.extend(y)

            all_preds = torch.stack(all_preds)
            all_targets = torch.stack(all_targets)
            base_path = "/".join(re.split("\\\|/", path)[:-1])
            with force_open(base_path + '/preds.pkl', 'wb') as file:
                pickle.dump(all_preds, file)
                logger.info(f'predictions are saved at {file.name}.')
            with force_open(base_path + '/targets.pkl', 'wb') as file:
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
        return str(self.init_lr)


class ANNModule(AbstractFeedForward):

    def __init__(self, dimension_list, dropout_list, *args, **kwargs):
        super(AbstractFeedForward, self).__init__(*args, **kwargs)

        if len(dropout_list) > len(dimension_list):
            raise ValueError(f"the length of dropout_list should be less equal than that of dimension_list: {len(dropout_list)} > {len(dimension_list)} ")
        
        self.dropout_list = [0] * len(dimension_list)
        self.dropout_list[:len(dropout_list)] = dropout_list
        
        self.dimension_list = dimension_list + [OUTPUT_LAYER_NODES]
        
        self.i2h = nn.Linear(self.input_size,
        self.dimension_list[0] if len(self.dimension_list) > 0 else OUTPUT_LAYER_NODES)
        torch.nn.init.normal_(self.i2h.weight)
        self.layers = nn.ModuleList()
        for i, j, d in zip(self.dimension_list, self.dimension_list[1:], self.dropout_list):
            l = nn.Linear(in_features=i, out_features=j)
            self.layers.append(nn.Dropout(d))
            torch.nn.init.normal_(l.weight)
            self.layers.append(l)
        # self.h2o = torch.nn.Linear(self.dimension_list[-1] if len(self.dimension_list) > 0 else input_size, OUTPUT_LAYER_NODES)
        # torch.nn.init.normal_(self.h2o.weight)

        logger.info(f"dimension list of nodes: {self.dimension_list}")
        logger.info(f"dropout list: {self.dropout_list}")

    @classmethod
    def short_name(cls) -> str:
        return "ann"

    def forward(self, x):
        x = self.i2h(x)
        for layer in self.layers:
            x = layer(self.activation(x))

        # x = self.h2o(x)
        # x = torch.sigmoid(x)
        # x = torch.clamp(x, min=1e-12, max=1 - 1e-12)
        return x

    def __str__(self) -> str:
        return str(self.init_lr) + "-" + ".".join((str(l) for l in self.dimension_list)) + "-" + ".".join((str(d) for d in self.dropout_list))

class SuperDynamicLossANN(ANNModule):

    def learn(self, epoch_num: int, batch_size: int, splits: list, train_dataset: Dataset, weights_checkpoint_path: str=None, condition_save_threshold=0.9):
        if weights_checkpoint_path is not None and len(weights_checkpoint_path):
            checkpoint = torch.load(weights_checkpoint_path)
            self.load_state_dict(checkpoint.get("model", checkpoint))

        logger.info(f"saving epoch condition: f2score>{condition_save_threshold}")
        logger.info("training phase started")
        scheduler_args = {"verbose":False, "min_lr":0, "threshold":1e-4, "patience":10, "factor":0.25}
        folds_metrics = []
        logger.info(f"number of folds: {len(splits)}")
        for fold, (train_ids, validation_ids) in enumerate(splits):
            self.train()
            logger.info("Resetting Optimizer, Learning rate, and Scheduler")
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.init_lr, momentum=0.9)
            last_lr = self.init_lr
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
            total_validation_loss = []
            # resetting module parameters
            self.reset_modules(module=self)
            fold_loss_function = DynamicSuperLoss(len(train_ids), 1, self.loss_function)
            validation_fold_loss_function = DynamicSuperLoss(len(validation_ids), 1, self.loss_function)
            for i in range(epoch_num):
                loss = 0
                epoch_loss = 0
                for batch_index, (X, y, items_indices) in enumerate(train_loader):
                    self.optimizer.zero_grad()
                    X = X.to(self.device)
                    y = y.to(self.device)
                    y = y.reshape(-1, 1).to(self.device)
                    y_hat = self.forward(X)
                    loss = fold_loss_function(y_hat, y, items_indices)
                    loss.backward()
                    self.optimizer.step()
                    logger.debug(f"fold: {fold} | epoch: {i} | batch: {batch_index} | loss: {loss}")
                    epoch_loss += loss.item()
                epoch_loss /= len(train_ids)
                total_loss.append(epoch_loss)
                self.scheduler.step(loss)
                if self.optimizer.param_groups[0]["lr"] != last_lr:
                    logger.info(f"fold: {fold} | epoch: {i} | Learning rate changed from: {last_lr} -> {self.optimizer.param_groups[0]['lr']}")
                    last_lr = self.optimizer.param_groups[0]["lr"]

                all_preds = []
                all_targets = []
                validation_loss = 0
                self.eval()
                with torch.no_grad():
                    for batch_index, (X, y, items_indices) in enumerate(validation_loader):
                        X = X.to(self.device)
                        y = y.to(self.device)
                        pred = self.forward(X).reshape(-1)
                        loss = validation_fold_loss_function(pred, y, items_indices)
                        validation_loss += loss
                        all_preds.extend(torch.sigmoid(pred) if isinstance(self.loss_function, nn.BCEWithLogitsLoss) else pred)
                        all_targets.extend(y)
                    validation_loss /= len(validation_ids)
                    total_validation_loss.append(validation_loss)
                all_preds = torch.stack(all_preds)
                all_targets = torch.stack(all_targets)
                
                accuracy_value, recall_value, precision_value, f2score, f05score = calculate_metrics_extended(all_preds, all_targets, device=self.device)

                logger.info(f"fold: {fold} | epoch: {i} | train -> loss: {(epoch_loss):>0.5f} | validation -> loss: {(validation_loss):>0.5f} | accuracy: {(100 * accuracy_value):>0.6f} | precision: {(100 * precision_value):>0.6f} | recall: {(100 * recall_value):>0.6f} | f2: {(100 * f2score):>0.6f} | f0.5: {(100 * f05score):>0.6f}")
                epoch_snapshot_path = self.get_detailed_session_path(train_dataset, "weights", f"f{fold}", f"model_f{fold}_e{i}.pth")
                if f2score >= condition_save_threshold:
                    logger.info(f"fold: {fold} | epoch: {i} | saving model at {epoch_snapshot_path}")
                    self.save(epoch_snapshot_path)
                self.train()

            folds_metrics.append((accuracy_value, precision_value, recall_value))
            
            snapshot_path = self.get_detailed_session_path(train_dataset, "weights", f"f{fold}", f"model_f{fold}.pth")
            self.save(snapshot_path)
            plt.clf()
            plt.plot(np.arange(1, 1 + len(total_loss)), np.array(total_loss), "-r", label="training")
            plt.plot(np.arange(1, 1 + len(total_loss)), np.array(total_validation_loss), "-b", label="validation")
            plt.legend()
            plt.title(f"fold #{fold}")
            with force_open(self.get_detailed_session_path(train_dataset, "figures", f"loss_f{fold}.png"), "wb") as f:
                plt.savefig(f, dpi=300)

    @classmethod
    def short_name(cls) -> str:
        return "ann-with-superloss"

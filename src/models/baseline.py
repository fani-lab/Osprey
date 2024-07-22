from typing import List, Any

import pickle
import logging
import shutil
from glob import glob
from csv import DictWriter
import re

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers.modeling_utils import PreTrainedModel

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from src.utils.commons import (RegisterableObject, roc_auc, calculate_metrics_extended, roc,
                                precision_recall_auc, precision_recall_curve, force_open)
import settings

logger = logging.getLogger()


class Baseline(RegisterableObject):

    def __init__(self, input_size: int, activation, loss_func, lr, module_session_path, validation_steps=-1,
                 device='cpu', early_stop=False, session_name="", do_aggeragate_metrics=True, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.init_lr = lr
        self.validation_steps = validation_steps
        self.activation = activation
        self.session_name = session_name
        self.do_aggeragate_metrics = do_aggeragate_metrics

        self.loss_function = loss_func

        self.session_path = module_session_path if module_session_path[-1] == "\\" or module_session_path[
            -1] == "/" else module_session_path + "/"

        self.snapshot_steps = 2
        self.device = device
        self.early_stop = early_stop

    def learn(self):
        raise NotImplementedError()

    def test(self):
        # load the saved model and apply it on test set
        # save the prediction results
        raise NotImplementedError()

    def get_session_path(self, *args):
        raise NotImplementedError()

    def check_stop_early(self, *args, **kwargs):
        return False
    
    def aggeregate(self, session_path, accuracies, recalls, precisions, f2scores, f05scores, aurocs, pr_aucs):
        fieldnames = ["session-name", "sessions-start-time", "model", "path", "notes", 'aucroc-avg', 'aucroc-std', 'aucroc-var', 'aucpr-avg', 'aucpr-std', 'aucpr-var', 'accurcay-avg', 'accurcay-std', 'accurcay-var', 'precision-avg', 'precision-std', 'precision-var', 'recall-avg', 'recall-std', 'recall-var', 'f2-avg', 'f2-std', 'f2-var', 'f05-avg', 'f05-std', 'f05-var', "logger_path"]
        
        with open(settings.AGGERAGETD_METRICS_PATH, "a") as f:
            writer = DictWriter(f, fieldnames=fieldnames, delimiter=',', lineterminator='\n')
            if f.tell() == 0:
                writer.writeheader()
            row = {"session-name": self.session_name, "sessions-start-time": settings.get_start_time(), "model": self.short_name(), "path": session_path, "notes": "",
                   'aucroc-avg': np.average(aurocs), 'aucroc-std': np.std(aurocs), 'aucroc-var': np.var(aurocs),
                   'aucpr-avg': np.average(pr_aucs), 'aucpr-std': np.std(pr_aucs), 'aucpr-var': np.var(pr_aucs),
                   'accurcay-avg': np.average(accuracies), 'accurcay-std': np.std(accuracies), 'accurcay-var': np.var(accuracies),
                   'precision-avg': np.average(precisions), 'precision-std': np.std(precisions), 'precision-var': np.var(precisions),
                   'recall-avg': np.average(recalls), 'recall-std': np.std(recalls), 'recall-var': np.var(recalls),
                   'f2-avg': np.average(f2scores), 'f2-std': np.std(f2scores), 'f2-var': np.var(f2scores),
                   'f05-avg': np.average(f05scores), 'f05-std': np.std(f05scores), 'f05-var': np.var(f05scores),
                   }
            log_filehandlers_path = [i.baseFilename for i in logger.handlers if hasattr(i, 'baseFilename')]
            row["logger_path"] = log_filehandlers_path[0]
            writer.writerow(row)

    def evaluate(self, path, device):
        folds = glob(path + "/weights/f*")
        logger.info(f"found #{len(folds)} folds at path: {path}")
        folds_accuracy, folds_recall, folds_precision, folds_f2score, folds_f05score, folds_auroc, folds_pr_auc = [0.0]*len(folds), [0.0]*len(folds), [0.0]*len(folds), [0.0]*len(folds), [0.0]*len(folds), [0.0]*len(folds), [0.0]*len(folds)
        for fold_num, fold_path in enumerate(folds):
            preds = None
            targets = None
            with open(fold_path+'/preds.pkl', 'rb') as file:
                preds = pickle.load(file)
            with open(fold_path+'/targets.pkl', 'rb') as file:
                targets = pickle.load(file)
                targets = targets.to(device)
            if preds.ndim > targets.ndim:
                preds = preds.reshape(-1)
            preds = preds.to(device)
            # targets = torch.argmax(targets, dim=1)
            fpr, tpr, _ = roc(preds, targets, device=device)
            auroc = roc_auc(preds, targets, device=device)

            roc_path = fold_path + "/ROC-curve.png"
            precision_recall_path = fold_path + "/precision-recall-curve.png"
            plt.clf()
            plt.plot(fpr.cpu(), tpr.cpu())
            plt.title("ROC")
            plt.savefig(roc_path)
            logger.info(f"saving ROC curve at: {roc_path}")
            precisions, recalls, _ = precision_recall_curve(preds, targets)
            pr_auc = precision_recall_auc(preds, targets, device=device)
            plt.clf()
            plt.plot(recalls.cpu(), precisions.cpu())
            plt.title("Recall-Precision Curve")
            plt.savefig(precision_recall_path)
            logger.info(f"saving precision-recall curve at: {precision_recall_path}")
            # plt.show()
            accuracy, recall, precision, f2score, f05score = calculate_metrics_extended(preds, targets, device=device)
            folds_accuracy[fold_num] = float(accuracy)
            folds_recall[fold_num] = float(recall)
            folds_precision[fold_num] = float(precision)
            folds_f2score[fold_num] = float(f2score)
            folds_f05score[fold_num] = float(f05score)
            folds_auroc[fold_num] = float(auroc)
            folds_pr_auc[fold_num] = float(pr_auc)
            logger.info(f"test set -> AUCROC: {(auroc):>0.7f} | AUCPR: {(pr_auc):>0.7f} | accuracy: {(accuracy):>0.7f} | precision: {(precision):>0.7f} | recall: {(recall):>0.7f} | f2score: {(f2score):>0.7f} | f0.5: {(100 * f05score):>0.6f}")
        
        logger.info(f"avg test set -> AUCROC: {(np.average(folds_auroc)):>0.7f} | AUCPR: {(np.average(folds_pr_auc)):>0.7f} | accuracy: {(np.average(folds_accuracy)):>0.7f} | precision: {(np.average(folds_precision)):>0.7f} | recall: {(np.average(folds_recall)):>0.7f} | f2score: {(100 * np.average(folds_f2score)):>0.7f} | f0.5: {(100 * np.average(folds_f05score)):>0.6f}")

        if self.do_aggeragate_metrics:
            self.aggeregate(session_path=path, accuracies=folds_accuracy, recalls=folds_recall, precisions=folds_precision, f2scores=folds_f2score, f05scores=folds_f05score, aurocs=folds_auroc, pr_aucs=folds_pr_auc)


class PytorchBaseline(Baseline, nn.Module):

    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        Baseline.__init__(self, *args, **kwargs)

    def forward(self, x: Any) -> torch.Tensor:
        raise NotImplementedError()
    
    def get_test_dataloaders(self, dataset: Dataset, batch_size: int) -> DataLoader:
        raise NotImplementedError()

    def get_detailed_session_path(self, dataset: Dataset, *args):
        details = str(dataset) + "-" + str(self)
        return self.get_session_path(details, *args)
    
    def get_new_optimizer(self, lr: float, *args, **kwargs):
        raise NotImplementedError()
    
    def get_new_scheduler(self, optimizer: torch.optim.Optimizer, *args, **kwargs):
        raise NotImplementedError()
    
    def get_dataloaders(self, dataset: Dataset, train_ids: List[int], validation_ids: List[int], batch_size: int):
        raise NotImplementedError()
    
    def get_all_folds_checkpoints(self, dataset: Dataset):
        raise NotImplementedError()
    
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

    # add functionality for validation on test_dataset
    def learn(self, epoch_num: int, batch_size: int, splits: list, train_dataset: Dataset, weights_checkpoint_path: str=None, condition_save_threshold=0.9):
        if weights_checkpoint_path is not None and len(weights_checkpoint_path):
            checkpoint = torch.load(weights_checkpoint_path)
            self.load_state_dict(checkpoint.get("model", checkpoint))

        logger.info(f"saving epoch condition: f2score>{condition_save_threshold}")
        logger.info("training phase started")
        
        folds_metrics = []
        logger.info(f"number of folds: {len(splits)}")
        # todo calculate time spent for each fold or epoch or something, for convenience

        for fold, (train_ids, validation_ids) in enumerate(splits):
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

            for i in range(1, epoch_num + 1):
                self.train()
                loss = 0
                epoch_loss = 0
                if self.optimizer.param_groups[0]["lr"] != last_lr:
                    logger.info(f"fold: {fold} | epoch: {i} | Learning rate changed from: {last_lr} -> {self.optimizer.param_groups[0]['lr']}")
                    last_lr = self.optimizer.param_groups[0]["lr"]
                
                for batch_index, (X, y) in enumerate(tqdm(train_loader, leave=False)):
                    self.optimizer.zero_grad()
                    if isinstance(X, tuple) or isinstance(X, list):
                        X = [l.to(self.device) for l in X]
                    else:
                        X = X.to(self.device)
                    y_hat = self.forward(X)
                    y = y.reshape(-1).to(self.device)
                    y_hat = y_hat.reshape(-1).to(self.device)
                    loss = self.loss_function(y_hat, y)
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
                        if isinstance(X, tuple) or isinstance(X, list):
                            X = [l.to(self.device) for l in X]
                        else:
                            X = X.to(self.device)
                        pred = self.forward(X)
                        y = y.reshape(-1).to(self.device)
                        pred = pred.reshape(-1).to(self.device)
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
                self.scheduler.step(validation_loss)
                epoch_snapshot_path = self.get_detailed_session_path(train_dataset, "weights", f"f{fold}", f"model_f{fold}_e{i}.pth")
                if f2score >= condition_save_threshold:
                    logger.info(f"fold: {fold} | epoch: {i} | saving model at {epoch_snapshot_path}")
                    self.save(epoch_snapshot_path)
                
                if self.check_stop_early(f2score=f2score):
                    logger.info(f"early stop condition satisfied: f2 score => {f2score}")
                    break
            folds_metrics.append((accuracy_value, precision_value, recall_value, f2score))
            snapshot_path = self.get_detailed_session_path(train_dataset, "weights", f"f{fold}", f"model_f{fold}.pth")
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
        best_model_src = self.get_detailed_session_path(train_dataset, "weights", f"f{max_metric[0]}", f"model_f{max_metric[0]}.pth")
        shutil.copyfile(best_model_src, best_model_dest)
    
    def test(self, test_dataset, weights_checkpoint_path):
        for path in weights_checkpoint_path:
            logger.info(f"testing checkpoint at: {path}")
            torch.cuda.empty_cache()
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint.get("model", checkpoint))

            all_preds = []
            all_targets = []
            test_dataloader = self.get_test_dataloaders(test_dataset, 64)
            self.eval()
            with torch.no_grad():
                for X, y in test_dataloader:
                    X = X.to(self.device)
                    y_hat = self.forward(X)
                    y = y.reshape(-1).to(self.device)
                    y_hat = y_hat.reshape(-1).to(self.device)
                    all_preds.extend(torch.sigmoid(y_hat) if isinstance(self.loss_function, nn.BCEWithLogitsLoss) else y_hat)
                    all_targets.extend(y)

            all_preds = torch.tensor(all_preds)
            all_targets = torch.tensor(all_targets)
            base_path = "/".join(re.split("\\\|/", path)[:-1])
            with force_open(base_path + '/preds.pkl', 'wb') as file:
                pickle.dump(all_preds, file)
                logger.info(f'predictions are saved at: {file.name}')
            with force_open(base_path + '/targets.pkl', 'wb') as file:
                pickle.dump(all_targets, file)
                logger.info(f'targets are saved at: {file.name}')


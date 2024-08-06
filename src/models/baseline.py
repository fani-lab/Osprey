import pickle
import logging
from glob import glob
from csv import DictWriter
import re

import numpy as np
import matplotlib.pyplot as plt

from src.utils.commons import RegisterableObject, roc_auc, calculate_metrics_extended, roc, precision_recall_auc, precision_recall_curve
import settings

logger = logging.getLogger()

__MAX_NUMBER_OF_FOLDS__ = 10

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
    
    def get_new_optimizer(self, lr, *args, **kwargs):
        raise NotImplementedError()
    
    def get_new_scheduler(self, optimizer, *args, **kwargs):
        raise NotImplementedError()
    
    def get_all_folds_checkpoints(self, dataset):
        raise NotImplementedError()
    
    def check_stop_early(self, *args, **kwargs):
        return False
    
    def aggeregate(self, session_path, accuracies, recalls, precisions, f2scores, f05scores, aurocs, pr_aucs, notes=""):
        fieldnames = ["session-name", "sessions-start-time", "model", "path", "notes", 'aucroc-avg', 'aucroc-std', 'aucroc-var', 'aucpr-avg', 'aucpr-std', 'aucpr-var', 'accurcay-avg', 'accurcay-std', 'accurcay-var', 'precision-avg', 'precision-std', 'precision-var', 'recall-avg', 'recall-std', 'recall-var', 'f2-avg', 'f2-std', 'f2-var', 'f05-avg', 'f05-std', 'f05-var', "logger_path"]
        
        with open(settings.AGGERAGETD_METRICS_PATH, "a") as f:
            writer = DictWriter(f, fieldnames=fieldnames, delimiter=',', lineterminator='\n')
            if f.tell() == 0:
                writer.writeheader()
            row = {"session-name": self.session_name, "sessions-start-time": settings.get_start_time(), "model": self.short_name(), "path": session_path, "notes": notes,
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
    
    def evaluate(self, path, device='cpu'):
        preds_pattern = re.compile(".*(preds|targets)\\.pkl$")
        all_folds = [p.replace('\\', '/') for p in glob(path + "/weights/f*/**", recursive=True)]
        tests_with_folds = dict()
        for path in all_folds:
            if not preds_pattern.match(path):
                continue
            fold_num, *dataset_name = path.split('weights/')[-1].split('/')
            fold_num = int(fold_num[1:])
            dataset_name = '/'.join(dataset_name[:-1])
            temp = tests_with_folds.get(dataset_name, [None]* __MAX_NUMBER_OF_FOLDS__)
            temp[fold_num] = '/'.join(path.split('/')[:-1])
            tests_with_folds[dataset_name] = temp

        for dataset_name, folds in tests_with_folds.items():
            folds = [ff for ff in folds if ff is not None]
            logger.info(f'evaluating test of dataset: {dataset_name}')
            logger.info(f"found #{len(folds)} folds at path: {folds}")
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
                
                accuracy, recall, precision, f2score, f05score = calculate_metrics_extended(preds, targets, device=device)
                folds_accuracy[fold_num] = float(accuracy)
                folds_recall[fold_num] = float(recall)
                folds_precision[fold_num] = float(precision)
                folds_f2score[fold_num] = float(f2score)
                folds_f05score[fold_num] = float(f05score)
                folds_auroc[fold_num] = float(auroc)
                folds_pr_auc[fold_num] = float(pr_auc)
                logger.info(f"test set -> AUCROC: {(auroc):>0.6f} | AUCPR: {(pr_auc):>0.6f} | accuracy: {(accuracy):>0.6f} | precision: {(precision):>0.6f} | recall: {(recall):>0.6f} | f2score: {(f2score):>0.6f} | f0.5: {(f05score):>0.6f}")
            
            logger.info(f"avg test set -> AUCROC: {(np.average(folds_auroc)):>0.6f} | AUCPR: {(np.average(folds_pr_auc)):>0.6f} | accuracy: {(np.average(folds_accuracy)):>0.6f} | precision: {(np.average(folds_precision)):>0.6f} | recall: {(np.average(folds_recall)):>0.6f} | f2score: {(np.average(folds_f2score)):>0.6f} | f0.5: {(np.average(folds_f05score)):>0.6f}")
            notes = fold_path.split('/')[-1]
            if self.do_aggeragate_metrics: # for session path, just take the last fold path, it's alright
                self.aggeregate(session_path=fold_path, accuracies=folds_accuracy, recalls=folds_recall, precisions=folds_precision,
                                f2scores=folds_f2score, f05scores=folds_f05score, aurocs=folds_auroc, pr_aucs=folds_pr_auc, notes=notes)
        
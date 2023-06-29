import pickle
import logging
import re
from glob import glob

import matplotlib.pyplot as plt
from src.utils.commons import RegisterableObject, roc_auc, calculate_metrics_extended, roc, precision_recall_auc, precision_recall_curve


logger = logging.getLogger()


class Baseline(RegisterableObject):

    def __init__(self, input_size: int, activation, loss_func, lr, module_session_path, validation_steps=-1,
                 device='cpu', **kwargs):
        super().__init__()
        self.input_size = input_size
        self.init_lr = lr
        self.validation_steps = validation_steps
        self.activation = activation

        self.loss_function = loss_func

        self.session_path = module_session_path if module_session_path[-1] == "\\" or module_session_path[
            -1] == "/" else module_session_path + "/"

        self.snapshot_steps = 2
        self.device = device

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
    
    def evaluate(self, path, device):
        folds = glob(path + "/weights/f*")
        logger.info(f"found #{len(folds)} folds at path: {path}")
        average_accuracy, average_recall, average_precision, average_f2score, average_f05score, average_auroc, average_pr_auc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for fold_path in folds:
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
            average_accuracy += accuracy
            average_recall += recall
            average_precision += precision
            average_f2score += f2score
            average_f05score += f05score
            average_auroc += auroc
            average_pr_auc += pr_auc
            logger.info(f"test set -> AUCROC: {(auroc):>0.7f} | AUCPR: {(pr_auc):>0.7f} | accuracy: {(accuracy):>0.7f} | precision: {(precision):>0.7f} | recall: {(recall):>0.7f} | f2score: {(f2score):>0.7f} | f0.5: {(100 * f05score):>0.6f}")
        
        number_of_folds = len(folds)
        average_accuracy /= number_of_folds
        average_recall /= number_of_folds
        average_precision /= number_of_folds
        average_f2score /= number_of_folds
        average_f05score /= number_of_folds
        average_auroc /= number_of_folds
        average_pr_auc /= number_of_folds
        
        logger.info(f"avg test set -> AUCROC: {(average_auroc):>0.7f} | AUCPR: {(average_pr_auc):>0.7f} | accuracy: {(average_accuracy):>0.7f} | precision: {(average_precision):>0.7f} | recall: {(average_recall):>0.7f} | f2score: {(average_f2score):>0.7f} | f0.5: {(100 * average_f05score):>0.6f}")
        
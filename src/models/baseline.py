import pickle
import logging
import matplotlib.pyplot as plt
import torchmetrics
import torch

from src.utils.commons import RegisterableObject


logger = logging.getLogger()


class Baseline(RegisterableObject):
    def __init__(self, input_size: int):
        self.input_size = input_size

    def learn(self):
        raise NotImplementedError()

    def test(self):
        # load the saved model and apply it on test set
        # save the prediction results
        raise NotImplementedError()

    def get_session_path(self, *args):
        raise NotImplementedError()

    def eval(self, path, device):
        accuracy = torchmetrics.Accuracy('multiclass', num_classes=2, top_k=1).to(device)
        precision = torchmetrics.Precision('multiclass', num_classes=2, top_k=1).to(device)
        recall = torchmetrics.Recall('multiclass', num_classes=2, top_k=1).to(device)
        roc = torchmetrics.ROC(task="multiclass", num_classes=2).to(device)
        auroc = torchmetrics.AUROC(task="multiclass", num_classes=2).to(device)
        preds = None
        targets = None
        with open(path+'/preds.pkl', 'rb') as file:
            preds = pickle.load(file)
        with open(path+'/targets.pkl', 'rb') as file:
            targets = pickle.load(file)
        if preds.ndim > targets.ndim:
            preds = preds.squeeze()
        preds = preds.to(device)
        targets = torch.argmax(targets.to(device), dim=1)
        fpr, tpr, thresholds = roc(preds, targets)
        plt.clf()
        plt.plot(fpr[1].cpu(), tpr[1].cpu())
        plt.title("ROC")
        plt.savefig(path + "ROC.png")
        # plt.show()

        logger.info('Evaluation:')
        logger.info(f'torchmetrics Accuracy: {(100 * accuracy(preds, targets)):>0.1f}')
        logger.info(f'torchmetrics precision: {(100 * precision(preds, targets)):>0.1f}')
        logger.info(f'torchmetrics Recall: {(100 * recall(preds, targets)):>0.1f}')
        logger.info(f'torchmetrics AUCROC: {(auroc(preds, targets)):>0.1f}')

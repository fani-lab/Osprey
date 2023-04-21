import pickle
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import auc

from src.utils.commons import RegisterableObject, roc_auc, calculate_metrics


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

    def evaluate(self, path, device):

        preds = None
        targets = None
        with open(path+'/preds.pkl', 'rb') as file:
            preds = pickle.load(file)
        with open(path+'/targets.pkl', 'rb') as file:
            targets = pickle.load(file)
        if preds.ndim > targets.ndim:
            preds = preds.squeeze()
        preds = preds
        fpr, tpr, _ = roc_auc(preds, targets)
        auroc = auc(fpr, tpr)
        roc_path = path + "/ROC.png"
        plt.clf()
        plt.plot(fpr.cpu(), tpr.cpu())
        plt.title("ROC")
        plt.savefig(roc_path)
        logger.info(f"saving ROC curve at: {roc_path}")
        # plt.show()
        accuracy, precision, recall = calculate_metrics(preds, targets)
        logger.info(f"test set -> AUCROC: {(auroc):>0.7f} | accuracy: {(accuracy):>0.7f} | precision: {(precision):>0.7f} | recall: {(recall):>0.7f}")

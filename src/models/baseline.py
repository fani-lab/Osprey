import pickle
import logging
import matplotlib.pyplot as plt
import torchmetrics
from sklearn import metrics

logger = logging.getLogger()


class Baseline():
    def __init__(self):
        pass

    def prep(self):
        # if the features are saved => load it
        # else:
        #   extract
        #   save
        pass

    def learn(self):
        pass

    def test(self):
        # load the saved model and apply it on test set
        # save the prediction results
        pass

    def get_session_path(self, *args):
        raise NotImplementedError()

    def eval(self):
        accuracy = torchmetrics.Accuracy('binary', )
        precision = torchmetrics.Precision('binary', )
        recall = torchmetrics.Recall('binary', )
        roc = torchmetrics.ROC(task="binary")
        auroc = torchmetrics.AUROC(task="binary")
        preds = None
        targets = None
        with open(self.get_session_path('preds.pkl'), 'rb') as file:
            preds = pickle.load(file)
        with open(self.get_session_path('targets.pkl'), 'rb') as file:
            targets = pickle.load(file)
        fpr, tpr, thresholds = roc(preds, targets)
        plt.plot(fpr, tpr)
        plt.title("ROC")
        plt.savefig(self.get_session_path(f"ROC.png"))
        plt.show()

        logger.info('Evaluation:')
        logger.info(f'torchmetrics Accuracy: {(100 * accuracy(preds, targets)):>0.1f}')
        logger.info(f'torchmetrics precision: {(100 * precision(preds, targets)):>0.1f}')
        logger.info(f'torchmetrics Recall: {(100 * recall(preds, targets)):>0.1f}')
        logger.info(f'torchmetrics AUCROC: {(auroc(preds, targets)):>0.1f}')


    def main(self):
        # call the pipeline or part of it for prep, train, test, eval
        pass

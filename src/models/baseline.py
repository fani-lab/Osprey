import pickle
import logging

import torchmetrics

logger = logging.getLogger()


class Baseline():
    def __init__(self):
        pass

    def prep(self):
        # if the features are save => load it
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
        preds = None
        targets = None
        with open(self.get_session_path('preds.pkl')) as file:
            preds = pickle.load(file)
        with open(self.get_session_path('targets.pkl')) as file:
            targets = pickle.load(file)

        logger.info(f'torchmetrics Accuracy: {(100 * accuracy(preds, targets)):>0.1f}')
        logger.info(f'torchmetrics precision: {(100 * precision(preds, targets)):>0.1f}')
        logger.info(f'torchmetrics Recall: {(100 * recall(preds, targets)):>0.1f}')

    def main(self):
        # call the pipeline or part of it for prep, train, test, eval
        pass

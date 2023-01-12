import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import warnings


class Baseline:
    def __init__(self, features, output, split, target):
        self.rf = RandomForestClassifier()
        self.features = features
        self.split = split
        self.target = target
        self.output = output

    def prep(self):
        # splitting the train and test section of the csr_matrx
        rf_train = self.features[:self.split[0], :]  # [70%]
        rf_test = self.features[self.split[0]:, :]  # [30%]
        # return a list containing two csr_mtx for train and test
        return rf_train, rf_test

    def train(self, rf_train):

        self.rf.fit(rf_train, self.target[:self.split[0]].values.ravel())
        joblib.dump(self.rf, f"../output/rf/{self.output}.pkl", compress=3)

    def test(self, rf_test):
        # load the saved model and apply it on test set
        loaded_rf = joblib.load(f"../output/rf/{self.output}.pkl")
        pred_label = loaded_rf.predict(rf_test)
        joblib.dump(pred_label, f"../output/rf/{self.output}.pred.test.pkl", compress=3)
        return pred_label
        # save the prediction results
        # self.rf.pred_prob()

    def eval(self):
        # load the prediction results
        # eval on test labels
        # save the eval results self.target[-self.split[1]]
        pass

    def main(self):  # rf_train, rf_test
        # call the pipeline or part of it for prep, train, test, eval
        warnings.filterwarnings('ignore', category=FutureWarning)

        lst = self.prep()
        rf_train = lst[0]
        rf_test = lst[1]

        self.train(rf_train)

        self.test(rf_test)

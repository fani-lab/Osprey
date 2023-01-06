from sklearn.ensemble import RandomForestClassifier


class Baseline:
    def __init__(self, features, target):
        self.rf = RandomForestClassifier()
        self.features = features
        self.target = target

    def prep(self):
        '''
        remove redundant/ useless features
        check if there are null values
        get rid of id 
        Returns
        -------

        '''
        pass

    def train(self, rf_train):
        label = rf_train['tagged_predator']
        self.rf.fit(rf_train, label)

        # using gridsearch cvjust to see if we get result:
    def train(self, rf_train):
        label = rf_train['tagged_pred']
        self.rf.fit(rf_train['', ''], label )

        return None
>>>>>>> Stashed changes

    def test(self):
        # load the saved model and apply it on test set
        # save the prediction results
        # self.rf.pred_prob()
        pass

    def eval(self):
        # load the prediction results
        # eval on test labels
        # save the eval results
        pass

    def main(self, rf_train, rf_test):
        # call the pipeline or part of it for prep, train, test, eval
        # prep
        self.prep()
        self.train(rf_train)  # rf: random_forest train features

        # self.test(rf_test)

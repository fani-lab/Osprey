from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

class Baseline:
    def __init__(self, features, target):
        self.rf = RandomForestClassifier()
        self.features = features
        self.target = target

    def prep(self, rf_train):
        '''
        remove redundant/ useless features
        check if there are null values
        get rid of id 
        Returns
        -------
        '''
        drop_list = ['conv_id', 'msg_line', 'author_id']
        rf_train.drop(drop_list, axis=1, inplace=True)

        # checking for missing values:
        print(f"{rf_train.count()}")

        # have to conver the string to time:
        datetime.strptime(rf_train['time'], '%A')

    def train(self, rf_train):

        print(f"Enter the number associated with the target of choice:\n"
              f"1. tagged_msg \n 2. tagged_predator \n 3. tagged_conv")
        choice = int(input())
        if choice == 1:
            label = self.target[0]
        if choice == 2:
            label = self.target[1]
        if choice == 3:
            label = self.target[2]
        else:
            print(f"invalid selection! Rerun the program")
            exit(1)

        self.rf.fit(rf_train, label)

        # using gridsearch cvjust to see if we get result:

    def test(self, rf_test):
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
        self.prep(rf_train)
        #self.train(rf_train)  # rf: random_forest train features
        #self.test(rf_test)


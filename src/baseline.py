import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class Baseline:
    def __init__(self, features, target, split):
        self.rf = RandomForestClassifier()
        self.features = features
        self.target = target
        self.split = split

    def prep(self):
        # getting the target/label from the user
        # print(f"Enter the number associated with the target of choice:\n"
        #       f"1. tagged_msg \n 2. tagged_predator\n 3. tagged_conv")
        # while True:
        #     choice = input()
        #     if choice == '1':
        #         label = self.target[0]
        #     elif choice == '2':
        #         label = self.target[1]
        #     elif choice == '3':
        #         label = self.target[2]
        #     else:
        #         print(f"invalid selection! retry by using the available options!")
        #         continue
        #     break

        # this is the concatenated that was used to build a classifier in 'main.py'
        #is this okay?
        rf_train_test = self.features

        # checking for missing values:
        #print(f"{rf_train_test.count()}")
        # convert the time aa string to time and then numeric format:

        #self.features['time'] = pd.to_numeric(pd.to_datetime(self.features['time']))



        #How to get the split here
        rf_train = self.features[:self.split[0], :]  # [70%]
        rf_test = self.features[self.split[1]:, :]  # [30%]
        # what happens to the validation portion?!

        # remove redundant/ useless features @test_drop_list
        train_drop_list = ['conv_id', 'msg_line', 'author_id', 'time']
        rf_train.drop(train_drop_list, axis=1, inplace=True)

        # remove redundant/ useless features @test_drop_list including the target/label
        test_drop_list = ['conv_id', 'msg_line', 'author_id', 'time', self.target]
        rf_test.drop(test_drop_list, axis=1, inplace=True)

        # Returns the data frames for train and test and the label we're training on as a list
        return [rf_train, rf_test]

    def train(self, rf_train):
        self.rf.fit(rf_train, rf_train[self.target])
        joblib.dump(rf_train, f"./output/rf/{self.target}.pkl")

    def test(self, rf_test, label):
        # load the saved model and apply it on test set
        # save the prediction results
        # self.rf.pred_prob()
        pass

    def eval(self):
        # load the prediction results
        # eval on test labels
        # save the eval results
        pass

    def main(self):  # rf_train, rf_test
        # call the pipeline or part of it for prep, train, test, eval

        lst = self.prep()
        rf_train = lst[0]
        rf_test = lst[1]

        # self.train(rf_train)
        # self.test(rf_test)

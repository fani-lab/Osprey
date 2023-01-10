from baseline import Baseline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
import pickle
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from lib import eval as evl
from sklearn.metrics import f1_score, precision_score, recall_score


class msg_classifier(Baseline):
    def __init__(self):#features, Q,  relabeling, df_train_test, split = [70, 30], target=['tagged_msg']):
        super(msg_classifier, self).__init__()
        #print(features)

    def prep(self):
        # already done in main.py
        pass 

    def train(self, X_train, y_train):
        model = LogisticRegression(solver='lbfgs', max_iter=1000)
        model.fit(X_train, y_train)
        pickle.dump(model, open("model.joblib", 'wb'))
        return model
        
    def test(self, X_test, y_test, model):
        #accuracy = model.score(X_test, y_test)
        #print("Accuracy: %.2f%%" % (accuracy * 100.0))
        return model.predict(X_test)

    def eval(self, targets, pred, model):
        targets = targets.values.flatten()

        #for tgt, pred in zip(targets, pred):
        #    print(tgt, pred)
        #    df = df.append(pd.DataFrame.from_dict(evl.evaluate(["F1_weighted"], tgt, pred)))
        
        #df.append(pd.DataFrame.from_dict({"f1":f1_score(targets, pred, average='weighted')}, index=[0]))
        df = pd.DataFrame([{"f1":f1_score(targets, pred, average='weighted'), "precision":precision_score(targets, pred, average='weighted'), "recall":recall_score(targets, pred, average='weighted')}])

        df.to_csv('preds.eval.csv',sep='\t')
        return df, df.mean()

    def main(self, df_train_test, X, cmd=['train', 'test', 'eval']):
        ROS = RandomOverSampler(sampling_strategy=1)
        y = df_train_test['tagged_msg']
        X,y = ROS.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        if 'train' in cmd: model = self.train(X_train, y_train)
        if 'test'  in cmd: ypred = self.test(X_test, y_test, model)
        if 'eval'  in cmd:_, mean = self.eval(y_test, ypred, model)



class conv_msg_classifier(msg_classifier):
    def __init__(self):
        super(msg_classifier, self).__init__()
        conversation_feature_sets = [['line'], ['time'], ['n_msgs'], ['is_binconv']]
        pass
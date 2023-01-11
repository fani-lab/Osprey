from baseline import Baseline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
import pickle
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from lib import eval as evl

class msg_classifier(Baseline):
    """Classifies message as predatory or normal

    Args:
        Baseline (Class): inherits from parent
    """
    def __init__(self):
        """init function"""
        super(msg_classifier, self).__init__()

    def prep(self, X, df_train_test):
        # preps the data by balancing and splitting into train and test sets
        ROS = RandomOverSampler(sampling_strategy=1)
        y = df_train_test['tagged_msg']
        X,y = ROS.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        # trains the model
        model = LogisticRegression(solver='lbfgs', max_iter=1000)
        model.fit(X_train, y_train)
        pickle.dump(model, open("model.joblib", 'wb'))
        return model
        
    def test(self, X_test, model):
        # model predictions
        return model.predict(X_test)

    def eval(self, targets, pred, output):
        # evaluation on model predictions, outputs csv
        targets = targets.values.flatten()
        df = pd.DataFrame([{"f1":f1_score(targets, pred, average='weighted'), "precision":precision_score(targets, pred, average='weighted'), "recall":recall_score(targets, pred, average='weighted')}])
        df.to_csv(f'{output}preds.eval.csv', sep='\t', index=False)
        return df

    def main(self, df, text_features, output, cmd=['prep', 'train', 'test', 'eval']):
        if 'prep'  in cmd: X_train, X_test, y_train, y_test = self.prep(text_features, df)
        if 'train' in cmd: model = self.train(X_train, y_train)
        if 'test'  in cmd: ypred = self.test(X_test, model)
        if 'eval'  in cmd: result = self.eval(y_test, ypred, output)

class conv_msg_classifier(msg_classifier):
    def __init__(self):
        super(msg_classifier, self).__init__()
        conversation_feature_sets = [['line'], ['time'], ['n_msgs'], ['is_binconv']]
        pass
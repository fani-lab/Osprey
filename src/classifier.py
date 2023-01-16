from baseline import Baseline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler #download required
import pickle
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

class msg_classifier(Baseline):
    """Classifies message as predatory or normal

    Args:
        Baseline (Class): inherits from parent
    """
    def __init__(self):
        super(msg_classifier, self).__init__()

    def prep(self, X, df_train_test):
        # preps the data by balancing and splitting into train and test sets
        ROS = RandomOverSampler(sampling_strategy=1)
        y = df_train_test['tagged_msg']
        X,y = ROS.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train, output):
        # trains the model
        model = LogisticRegression(solver='lbfgs', max_iter=1000)
        model.fit(X_train, y_train)
        pickle.dump(model, open(f"{output}.joblib", 'wb'))
        return model
        
    def test(self, X_test, model):
        # model predictions
        return model.predict(X_test)

    def eval(self, targets, pred, output, feature_str):
        # evaluation on model predictions, outputs csv
        targets = targets.values.flatten()
        
        try:
            df = pd.read_csv("preds.eval.csv")
            new = pd.Dataframe([["features", "f1", "precision", "recall"], [feature_str, f1_score(targets, pred, average='weighted'), precision_score(targets, pred, average='weighted'), recall_score(targets, pred, average='weighted')]])
            df.update(new)
            df.to_csv(f'preds.eval.csv', sep='\t', index=False)

        except FileNotFoundError:
            print("File not found.")
            df = pd.Dataframe([["features", "f1", "precision", "recall"], [feature_str, f1_score(targets, pred, average='weighted'), precision_score(targets, pred, average='weighted'), recall_score(targets, pred, average='weighted')]])
            df.to_csv(f'preds.eval.csv', sep='\t', index=False)
        return df

    def main(self, df, text_features, output, feature_str, cmd=['prep', 'train', 'test', 'eval']):
        if 'prep'  in cmd: X_train, X_test, y_train, y_test = self.prep(text_features, df)
        if 'train' in cmd: model = self.train(X_train, y_train, output)
        if 'test'  in cmd: ypred = self.test(X_test, model)
        if 'eval'  in cmd: result = self.eval(y_test, ypred, output, feature_str)

class conv_msg_classifier(msg_classifier):
    def __init__(self):
        super(msg_classifier, self).__init__()
        conversation_feature_sets = [['line'], ['time'], ['n_msgs'], ['is_binconv']]
        pass
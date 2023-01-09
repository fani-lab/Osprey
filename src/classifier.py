from baseline import Baseline
from sklearn.model_selection import train_test_split

class msg_classifier(Baseline):
    def __init__(self, features, split = [70, 30], target=['tagged_msg']):
        super(msg_classifier, self).__init__()
        print(features)
    def prep(self):
        # already done in main.py
        pass 
    def train(self):
        pass
    def test(self):
        pass
    def eval(self):
        pass
    def main(self, X, Q, relabeling, df_train_test, cmd=['train', 'test', 'eval']):
        y = df_train_test['tagged_msg']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print(X_train, y)
        #if 'train' in cmd: model = self.train([X_train, y_train])
        #if 'test'  in cmd: ypred = self.test(X_test, model)
        #if 'eval'  in cmd:_, mean = self.eval(y_test, ypred)



class conv_msg_classifier(msg_classifier):
    def __init__(self):
        super(msg_classifier, self).__init__()
        conversation_feature_sets = [['line'], ['time'], ['n_msgs'], ['is_binconv']]
        pass
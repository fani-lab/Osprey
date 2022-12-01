from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
class Baseline():
    def __init__(self, features, target):
        self.rf = RandomForestClassifier()
        self.features = features 
        self.target = target
        
        self.parameters = {
            'n_estimators': [5, 50, 250],
            'max_depth': [2, 4, 8, 16, 32, None]
        } #There Paramters are usefull when using gridsearch cv
        
    def print_results(results):
        print('BEST PARAMS: {}\n'.format(results.best_params_))

        means = results.cv_results_['mean_test_score']
        stds = results.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, results.cv_results_['params']):
            print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))



    def prep(self):

        ##isn't this operation what is being done in main at line : 67 ? 
        #if the features are save => load it
        #else:
        #   extract
        #   save
        pass

    def train(self, rf_train):
        self.rf.fit(rf_train, self.target.ravel())

        #using gridsearch cvjust to see if we get result:
        cv = GridSearchCV(self.rf, self.parameters, cv=5)
        cv.fit(rf_train, self.target.ravel())
        self.print_results(cv) 

    def test(self):
        #load the saved model and apply it on test set
        #save the prediction results
        pass

    def eval(self):
        #load the prediction results
        #eval on test labels
        #save the eval results
        pass


    def main(self, rf_train, rf_test):
        #call the pipeline or part of it for prep, train, test, eval
        #prep
        self.train(rf_train) # rf: random_forest train features
        #self.test(rf_test) 
        
    


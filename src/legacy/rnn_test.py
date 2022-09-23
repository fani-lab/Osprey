import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn import svm, datasets
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from rnn_binary import get_train_test_data, Getvalues
# from tf_binary import get_train_test_data as gtt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
# import tensorflow as tf
from mrmr import mrmr_classif

tqdm.pandas(desc="progress-bar")
from sklearn.linear_model import LogisticRegression

dummy_clf = DummyClassifier(strategy="most_frequent")
from sklearn.manifold import TSNE

corpus_training_file = 'D:/data/train/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml'
corpus_training_predator_id_file = 'D:/data/train/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt'
dummy_training = 'D:/data/train/traindummy.xml'

corpus_test_file = 'D:/data/test/pan12-sexual-predator-identification-test-corpus-2012-05-17.xml'
corpus_test_predator_id_file = 'D:/data/test/pan12-sexual-predator-identification-groundtruth-problem1.txt'
# X_train, y_train = get_train_test_data('predators_only_train.xml', corpus_training_predator_id_file)
# X_test, y_test = get_train_test_data('predators_only_test.xml', corpus_test_predator_id_file)
# X_train2, y_train2 = gtt(dummy_training, corpus_training_predator_id_file)
# X_train, y_train = get_train_test_data('messagestats.xml', corpus_training_predator_id_file)

# X_test, y_test = get_train_test_data('predators_only_test.xml', corpus_test_predator_id_file)

X_train, y_train = datasets.load_iris(return_X_y=True)
# X_test = X_train
# y_test = y_train
#
# np.save('X_test.npy', X_test)
# np.save('y_test.npy', y_test)
# np.save('X_train_full.npy', X_train)
# np.save('y_train_full.npy', y_train)


# Load data
# X_train = np.load('X_train_full.npy')
# y_train = np.load('y_train_full.npy')
# X_test = np.load('X_test.npy')
# y_test = np.load('y_test.npy')

# selected_features = mrmr_classif(X_train, y_train, K=50)
# X_train = X_train[:, selected_features]

# pca = PCA(n_components=50)
# pca.fit(X_train)
# X_train = pca.transform(X_train)
#
# pca = PCA(n_components=50)
# pca.fit(X_test)
# X_test = pca.transform(X_test)

# print ("Transforming Train")
# X_train = TSNE(n_components=2, n_jobs=-1).fit_transform(X_train)
# print ("Transforming Test")
# X_test = TSNE(n_components=2, n_jobs=-1).fit_transform(X_test)

lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis(store_covariance=True)
gnb = GaussianNB()
neigh = KNeighborsClassifier(n_neighbors=15)
random_forest = RandomForestClassifier(max_depth=32)
logreg = LogisticRegression(n_jobs=1, C=1e5)
# svm()
cv = KFold(n_splits=10)

classifiers = [
    lda,
    # qda,
    # gnb,
    # neigh,
    # random_forest,
    # logreg
    # XGBClassifier(),
    dummy_clf
]

# model = tf.keras.Sequential([
#
#     tf.keras.layers.Embedding(
#         input_dim=len(encoder.get_vocabulary()),
#         output_dim=64,
#         # Use masking to handle the variable sequence lengths
#         mask_zero=True),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(1)
# ])


print("Classifiers about to start")
# for classifier in classifiers:
#     classifier.fit(X_train, y_train)
#     scores = classifier.predict(X_test)
#
#     conf_mat = confusion_matrix(y_test, scores)
#     print(conf_mat)
#     accuracy = accuracy_score(y_test, scores)
#     print(str(classifier) + ": ")
#     Sensitivity, Specificity, PPV, NPV = Getvalues(conf_mat)
#     print("Acuuracy : " + str(accuracy) + " Sensitivity : " + str(Sensitivity) +
#           " Specificity: " + str(Specificity) + " PPV: " + str(PPV) + " NPV: " + str(NPV))

for classifier in classifiers:
    scores = cross_val_predict(classifier, X_train, y_train, cv=cv)
    conf_mat = confusion_matrix(y_train, scores)
    accuracy = accuracy_score(y_train, scores)
    # fpr, tpr, thresholds = roc_curve(y_train, scores)
    # auc_score = auc(fpr, tpr)
    print(str(classifier) + " for ")
    # print(conf_mat)
    Sensitivity, Specificity, PPV, NPV = Getvalues(conf_mat)
    print("Acuuracy : " + str(accuracy) + " Sensitivity : " + str(Sensitivity) +
          " Specificity: " + str(Specificity) + " PPV: " + str(PPV) + " NPV: " + str(NPV)
          # , " AUC: " + str(auc_score)
    )



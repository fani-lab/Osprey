from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm

tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import re
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.test.utils import get_tmpfile

ROW_LENGTH = 5

dataset_train_dir = 'D:/data/train'
dataset_test_dir = 'D:/data/test'

corpus_training_file = 'D:/data/train/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml'
corpus_training_predator_id_file = 'D:/data/train/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt'
dummy_training = 'D:/data/train/traindummy.xml'

corpus_test_file = 'D:/data/test/pan12-sexual-predator-identification-test-corpus-2012-05-17.xml'
corpus_test_predator_id_file = 'D:/data/test/pan12-sexual-predator-identification-groundtruth-problem1.txt'

result_dir = '/data'
w2v = Word2Vec.load("word2vec.model")


def Getvalues(cm):
    TP = cm[0, 0]
    TN = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]

    Sensitivity = round(TP / (TP + FN), 2)
    Specificity = round(TN / (TN + FP), 2)
    PPV = round(TP / (TP + FP), 2)
    NPV = round(TN / (TN + FN), 2)
    F1_score = round(TP / (TP + 0.5 * (FP + FN)), 2)
    return Sensitivity, Specificity, PPV, NPV


def get_train_test_data(features_path, labels_path):
    predators = []
    file = open(labels_path, "r")
    for line in file:
        predators.append(line.strip())
    dict_list = []
    rows_list = []
    cols = ['author', 'time', 'text', 'ispredator', 'total_messages', 'message-length']
    tree = ET.parse(features_path)
    counter = 0
    row = dict()

    for elt in tree.iter():
        if elt.tag in cols:
            row[elt.tag] = elt.text
            if elt.tag == 'author':
                if elt.text in predators:
                    row['ispredator'] = 1
                else:
                    row['ispredator'] = 0

        if elt.tag == 'message':
            if row and row['text']:
                # row['len_of_message'] = len(row['text'])
                print(counter)
                counter = counter + 1
                # row['og_text'] = row['text']
                dict_list.append(row)
                row = dict()

    df = pd.DataFrame(dict_list)
    df = df[df['text'].notna()]
    df = df[df['ispredator'].notna()]

    # X = df['text'].to_numpy()
    # z = X
    # X = np.array(list(x for x in X))
    #
    # y = df['ispredator'].to_numpy()

    df = df[['text', 'ispredator']]
    df = df[pd.notnull(df['text'])]
    # df.rename(columns = {'Consumer complaint narrative':'narrative'}, inplace = True)
    df.index = range(df.shape[0])
    df['text'].apply(lambda x: len(x.split(' '))).sum()
    cnt_pro = df['ispredator'].value_counts()
    plt.figure(figsize=(8, 4))
    sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('ispredator', fontsize=12)
    plt.xticks(rotation=90)
    plt.show()

    def cleanText(text):
        text = BeautifulSoup(text, "lxml").text
        text = re.sub(r'\|\|\|', r' ', text)
        text = re.sub(r'http\S+', r'<URL>', text)
        text = text.lower()
        text = text.replace('x', '')
        return text

    df['text'] = df['text'].apply(cleanText)
    # train, test = train_test_split(df, test_size=0.3, random_state=42)
    import nltk
    from nltk.corpus import stopwords
    def tokenize_text(text):
        tokens = []
        for sent in nltk.sent_tokenize(text):
            for word in nltk.word_tokenize(sent):
                if len(word) < 2:
                    continue
                tokens.append(word.lower())
        return tokens

    train_tagged = df.apply(
        lambda r: TaggedDocument(words=tokenize_text(r['text']), tags=[r.ispredator]), axis=1)

    # model_dbow = Doc2Vec(dm=0, vector_size=100, negative=5, hs=0, min_count=2, sample=0, )
    # model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])
    #
    # for epoch in range(30):
    #     model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values),
    #                      epochs=1)
    #     model_dbow.alpha -= 0.002
    #     model_dbow.min_alpha = model_dbow.alpha

    # fname = get_tmpfile("my_doc2vec_model")
    # model_dbow.save('doc2vec.model')
    model_dbow = Doc2Vec.load('doc2vec.model')

    def vec_for_learning(model, tagged_docs):
        sents = tagged_docs.values
        targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
        return regressors, targets

    X_new = []
    X, y = vec_for_learning(model_dbow, train_tagged)
    for idx, value in enumerate(X):
        print(dict_list[idx]['total_messages'])
        value = np.append(value, dict_list[idx]['total_messages'])
        value = np.append(value, dict_list[idx]['message-length'])
        X_new.append(value)
    X = X_new
    return X, y

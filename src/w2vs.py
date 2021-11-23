import xml.etree.ElementTree as ET
# import gensim
import nltk as nltk
import pandas as pd
from gensim.models import Word2Vec
from sklearn import preprocessing

import numpy as np

# from random import random
# from numpy import array
# from numpy import cumsum
# from matplotlib import pyplot
# from pandas import DataFrame
from nltk import word_tokenize
from nltk.corpus import brown

dataset_train_dir = '../data/train'
dataset_test_dir = '../data/test'
#
corpus_training_file = '../data/train/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml'
corpus_training_predator_id_file = '../data/train/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt'
dummy_training = '../data/train/traindummy.xml'

corpus_test_file = '../data/test/pan12-sexual-predator-identification-test-corpus-2012-05-17.xml'
corpus_test_predator_id_file = '../data/test/pan12-sexual-predator-identification-groundtruth-problem1.txt'

result_dir = '../output'
# w2v = Word2Vec.load("word2vec.model")
predators = []
file = open(corpus_training_predator_id_file, "r")
for line in file:
    predators.append(line.strip())
dict_list = []
cols = ['author', 'time', 'text', 'ispredator']
# df = pd.DataFrame(columns=cols)
tree = ET.parse(corpus_training_file)
counter = 0
row = dict()
text = ''


def str2vec(str, model):
    tokens = str.split()
    vector = []
    print(str)
    for index, token in enumerate(tokens):
        single_token = []
        try:
            vec = model.wv.most_similar(token, topn=10)
            for v in vec:
                single_token.append(v[0])
            vector.append(single_token)
        except:
            pass
    print(vector)
    return vector


def tokenize(text):
    return word_tokenize(text)


for elt in tree.iter():
    if elt.tag in cols:
        row[elt.tag] = elt.text
        if elt.tag == 'author':
            if elt.text in predators:
                row['ispredator'] = 1
            else:
                row['ispredator'] = 0

    if elt.tag == 'message':
        # df = df.append(row, ignore_index=True)
        if row and row['text']:
            print(str(counter))
            print((row['text']))
            if len(row['text']) <= 300:#why such limit?
                row['text'] = tokenize(row['text'].lower())
                print('Tokenized')
            else:
                row['text'] = []
            # row['text'] = str2vec(row['text'], w2v)
            counter = counter + 1
            if row['text'] != []:
                dict_list.append(row)
                row = dict()

df = pd.DataFrame(dict_list)

df = df[df['author'].notna()]
df = df[df['time'].notna()]
df = df[df['text'].notna()]
df = df[df['ispredator'].notna()]

numerical_df = pd.DataFrame(columns=cols)

le = preprocessing.LabelEncoder()

X = []
y = []

numerical_df = numerical_df.dropna(subset=['ispredator'])
print("About to run save")

file_name = "../data/preprocessed/output.csv"
df.to_csv(file_name, sep='\t')
print('This ran')
#


text = df['text'].tolist()
print(text)
# print('This ran')

#

print("About to start tokenizing")
print("About to start training w2v")
w2v = Word2Vec(text, window=5, min_count=1, iter=100)#I would increase the #epochs to obtained the better vectors.
w2v.save("../output/word2vec.model")


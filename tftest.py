from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np

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
    return Sensitivity, Specificity, PPV, NPV, F1_score


def str2vec(str, model):
    tokens = str.split()
    vector = []
    # if len(tokens) == 1:
    for index, token in enumerate(tokens):
        single_token = []
        # expr1 if condition1 else expr2 if condition2 else expr
        try:
            w2v_default_value = model.wv[tokens[0]]
        except Exception:
            try:
                w2v_default_value = model.wv[tokens[1]]
            except:
                try:
                    w2v_default_value = model.wv[tokens[2]]
                except:
                    try:
                        w2v_default_value = model.wv[tokens[3]]
                    except:
                        try:
                            w2v_default_value = model.wv[tokens[4]]
                        except:
                            w2v_default_value = []
        except:
            w2v_default_value = []

        try:
            if token == '<et>':
                # vec = model.wv[tokens[0]]
                vec = w2v_default_value
            else:
                vec = model.wv[token]
            for v in vec:
                # single_token.append(v)
                vector.append(v)
        except:
            if len(w2v_default_value) != 0:
                for v in w2v_default_value:
                    vector.append(v)

    # vector = [item for sublist in vector for item in sublist]
    return vector


def get_multiple_rows(row):
    tokens = ' '.join(row['text'].split()).split()
    # tokens = row['text']
    n_word_string = ''
    list_of_rows = []
    for index, token in enumerate(tokens):
        if (index + 1) % ROW_LENGTH != 0:
            n_word_string = n_word_string + " " + token
        else:
            n_word_string = n_word_string + " " + token
            row['text'] = n_word_string
            list_of_rows.append(row.copy())
            n_word_string = ''
    if n_word_string != '':
        curr_length = len(n_word_string.strip().split(' '))
        if curr_length < ROW_LENGTH:
            for i in range(ROW_LENGTH - curr_length):
                n_word_string = n_word_string + ' <et>'
        row['text'] = n_word_string
        list_of_rows.append(row.copy())
    return list_of_rows


def get_train_test_data(features_path, labels_path):
    predators = []
    file = open(labels_path, "r")
    for line in file:
        predators.append(line.strip())
    dict_list = []
    rows_list = []
    cols = ['author', 'time', 'text', 'ispredator']
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
                row['og_text'] = row['text']
                row_length = len(row['text'].split())
                if row_length > ROW_LENGTH:
                    # continue
                    list_of_rows = get_multiple_rows(row)
                    for r in list_of_rows:
                        rows_list.append(r)
                elif row_length < ROW_LENGTH:
                    # continue
                    for i in range(ROW_LENGTH - row_length):
                        row['text'] = row['text'] + ' <et>'
                    rows_list.append(row)
                elif row_length == ROW_LENGTH:
                    rows_list.append(row)

                for row_to_append in rows_list:
                    row_to_append['text'] = str2vec(row_to_append['text'], w2v)
                    # This if taking a lot of time because of check !?
                    if len(row_to_append['text']) != 0:
                        del row_to_append['author']
                        del row_to_append['time']
                        dict_list.append(row_to_append)
                        row = dict()
                        counter = counter + 1
                        # Wo wala counter
                        print(str(counter))
        rows_list = []

    df = pd.DataFrame(dict_list)
    df = df[df['text'].notna()]
    df = df[df['ispredator'].notna()]
    # df.fillna(value='', inplace=True)
    # mlb = MultiLabelBinarizer()
    #
    # enc_df = pd.DataFrame(mlb.fit_transform(df.text),
    #                       columns=mlb.classes_,
    #                       index=df.text.index)
    X = df['text'].to_numpy()
    z = X
    X = np.array(list(x for x in X))
    # X = np.asarray(X).astype(np.float32)
    # X = np.array(df['text'], dtype=np.float)
    # X = np.asarray(X).astype(np.float32)
    y = df['ispredator'].to_numpy()

    for idx,ele in enumerate(z):
        if len(ele) != 500:
            print(len(ele), ele)
    return X, y

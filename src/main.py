import logging
import os
import sys
import time

import torch
from lxml import etree
import pandas as pd
# import extract_features as ef
# from classifier import msg_classifier
# from classifier import conv_msg_classifier
# import datetime
from models.ann import SimpleANN
from preprocessing.stopwords import NLTKStopWordRemoving
from preprocessing.punctuations import PunctuationRemoving
from utils.dataset import TimeBasedBagOfWordsDataset


def get_prev_msg_cat(prev, text):
    """Concatenates previous message with current message text

    Args:
        prev (dict): previous messages in conversation
        text (str): current message text

    Returns:
        str: past messages and current message text
    """
    return prev['prv_cat'] + " " + text


def get_next_msg_cat(conv, start_line, result):
    """Starts from message line and concats all future messages

    Args:
        conv (list): the entire conversation
        start_line (int): current message line number
        result (str): current message text

    Returns:
        str: "future messages" and current message text
    """
    if result is None:
        result = ""
    for msgs in conv[start_line:]:
        author, time, body = msgs.getchildren()
        if body.text is None:
            body.text = ""
        result += " " + body.text
    return result


def read_xml(xmlfile, tagged_msgs, predators):
    """Reads xml file to create dataset for training and testing sets

    Args:
        xmlfile (str): Conversation data to extract from.
        tagged_msgs (Dataframe): original labels (conversation, message id)
        predators (Dataframe): predator labels (author)

    Returns:
        Dataframe: dataset with text features and labels
    """
    dictionary_list = []
    df = pd.DataFrame(columns=['conv_id', 'msg_id', 'author_id', 'time', 'text'], index=['conv_id'])
    root = etree.parse(xmlfile).getroot()  # <conversations>
    for conv in root.getchildren():
        for msg in conv.getchildren():
            author, time, body = msg.getchildren()
            print(conv.get('id'))
            row = {'conv_id': conv.get('id'),
                   'msg_line': int(msg.get('line')),
                   'author_id': author.text,
                   'time': float(time.text.replace(":", ".")),
                   # previous messages in conversation & current message
                   # TODO Frightening space Complexity in following three lines
                   # 'prv_cat': "" if len(dictionary_list)==0 else get_prev_msg_cat(dictionary_list[-1], str(body.text)),
                   # # future messages in conversation & current message
                   # 'nxt_cat': get_next_msg_cat(conv.getchildren(), int(msg.get('line')), str(body.text)),
                   'msg_char_count': len(body.text) if body.text is not None else 0,
                   'msg_word_count': len(body.text.split()) if body.text is not None else 0,
                   'conv_size': len(conv.getchildren()),
                   # number of authors in the conversation, first index of msg
                   'nauthor': len(set([m.getchildren()[0].text for m in conv.getchildren()])),
                   'text': '' if body.text is None else body.text,
                   'tagged_msg': 0 if tagged_msgs.loc[(tagged_msgs['conv_id'] == conv.get('id')) & (
                           tagged_msgs['line'] == int(msg.get('line')))].empty else 1,
                   'tagged_conv': 0 if tagged_msgs.loc[tagged_msgs['conv_id'] == conv.get('id')].empty else 1,
                   'tagged_predator': None if predators.empty else (
                       1 if author.text in predators['tagged_pred'].tolist() else 0),
                   }
            dictionary_list.append(row)
    return df.from_dict(dictionary_list)


def get_stats(data):
    """_summary_

    Args:
        data: the input is dataframe

    Returns:
        _type_: _description_
    """

    stats = {'n_convs': len(data.groupby('conv_id')),
             'n_msgs': len(data),
             'avg_n_msgs_convs': round(len(data) / len(data.conv_id.unique()), 2),
             'n_binconv': len(data[data['nauthor'] == 2].groupby('conv_id')),
             'n_n-aryconv': len(data[data['nauthor'] > 2].groupby('conv_id')),
             'avg_n_msgs_binconvs': round(
                 len(data[data['nauthor'] == 2]) / len(data[data['nauthor'] == 2].groupby('conv_id')), 2),
             'avg_n_msgs_nonbinconvs': round(
                 len(data[data['nauthor'] > 2]) / len(data[data['nauthor'] > 2].groupby('conv_id')), 2),

             'n_tagged_binconvs': len(data[(data['nauthor'] == 2) & (data['tagged_conv'] == 1)].groupby('conv_id')),
             # needs relabeling: 1) any convs with at least one tagged_msg, 2) any convs with at least one predator
             'n_tagged_nonbinconvs': len(data[(data['nauthor'] > 2) & (data['tagged_conv'] == 1)].groupby('conv_id')),
             # needs relabeling
             'avg_n_msgs_tagged_convs': round(
                 len(data[data['tagged_conv'] == 1]) / len(data[data['tagged_conv'] == 1].groupby('conv_id')), 2),

             'n_convs_mult_predators': 0,
             'avg_n_msgs_convs_for_predator': round(len(data[(data['tagged_predator'] == 1)]) / len(
                 data[(data['tagged_predator'] == 1)].groupby('conv_id')), 2),
             'avg_n_normalconvs_for_predator': len(data[(data['tagged_conv'] == 0) & (data['tagged_predator'] == 1)]),
             }

    n_convs_mult_predators = 0
    for id, gp in data.groupby('conv_id'):
        num = 0
        for id1, authors in gp.groupby('author_id'):
            if authors.iloc[0].tagged_predator == 1:
                num = num + 1
        if num > 1:
            n_convs_mult_predators = n_convs_mult_predators + 1
    stats['n_convs_mult_predators'] = n_convs_mult_predators

    return stats


START_TIME = time.strftime("%m-%d-%Y-%H-%M-%S", time.localtime())

FORMATTER = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s : %(message)s")
FORMATTER_VERBOSE = logging.Formatter(
    "%(asctime)s | %(name)s | %(levelname)s | %(filename)s %(funcName)s @ %(lineno)s : %(message)s")

debug_file_handler = logging.FileHandler(f"logs/{START_TIME}.log")
debug_file_handler.setLevel(logging.DEBUG)
debug_file_handler.setFormatter(FORMATTER_VERBOSE)

info_terminal_handler = logging.StreamHandler(sys.stdout)
info_terminal_handler.setLevel(logging.INFO)
info_terminal_handler.setFormatter(FORMATTER)

logger = logging.getLogger()
logger.addHandler(debug_file_handler)
logger.addHandler(info_terminal_handler)
logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    # test_path, train_path = "/data/test/test.csv", "/data/train/train.csv"
    test_path, train_path = "data/toy.test/toy-test.csv", "data/toy.train/toy-train.csv"
    logger.info("reading test csv file")
    test_df = pd.read_csv(test_path)
    logger.info("reading train csv file")
    train_df = pd.read_csv(train_path)
    logger.debug("reading test and train csv files is done")

    datasets_path = "/data/preprocessed/ann/1/"

    train_dataset = TimeBasedBagOfWordsDataset(train_df, datasets_path, True,
                                      preprocessings=[NLTKStopWordRemoving(), PunctuationRemoving()], copy=False)
    train_dataset.prepare()
    test_dataset = TimeBasedBagOfWordsDataset(test_df, datasets_path + "test_", True,
                                     preprocessings=[NLTKStopWordRemoving(), PunctuationRemoving()],
                                     parent_dataset=train_dataset, copy=False)
    test_dataset.prepare()
    ## dimension_list, activation, loss_func, lr, train: pd.DataFrame, test: pd.DataFrame, preprocessings = list[BasePreprocessing], copy = True, load_from_pkl = True, preprocessed_path = "data/preprocessed/basic/"
    kwargs = {
        # "dimension_list": list([950, 250, 150, 50, 2]),
        "dimension_list": list([128]),
        "activation": torch.nn.ReLU(),
        "loss_func": torch.nn.CrossEntropyLoss(),
        "lr": 0.1,
        "train_dataset": train_dataset,
        "number_of_classes": 2,
    }
    model = SimpleANN(**kwargs)

    model.learn(epoch_num=50, batch_size=64)
    model.test(test_dataset)

# if __name__ == '__main__':
#     # test_path, train_path = "data/test/test.csv", "data/train/train.csv"
#     test_path, train_path = "../data/toy.test/toy-test.csv", "../data/toy.train/toy-train.csv"
#     kwargs = {
#         # "preprocessed_path": "data/preprocessed/basic/",
#         "preprocessed_path": "data/preprocessed/basic/toy-",
#         "load_from_pkl": False,
#         "preprocessings": [NLTKStopWordRemoving()],
#     }
#     test_df  = pd.read_csv(test_path)
#     train_df = pd.read_csv(train_path)
#
#     model = SimpleANN(train_df, test_df, **kwargs)
#
#     try:
#         model.prep()
#     except Exception as e:
#         e.with_traceback()
#
#     # HERE WE GO: -------------
#     model.train_tokens
#     model.train_labels
#
#     datapath = 'data/'
#
#     if os.path.isfile(f'{datapath}toy.train/toy-train.csv'):
#         df_train = pd.read_csv(f'{datapath}toy.train/toy-train.csv', index_col=0).fillna('')
#     else:
#         print("extracting training data")
#         training_file = f'{datapath}toy.train/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml'
#         training_predator_id_file = f'{datapath}toy.train/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt'
#         training_tagged_msgs_file = f'{datapath}toy.train/pan12-sexual-predator-identification-diff.txt'
#         df_train = read_xml(training_file, pd.read_csv(training_tagged_msgs_file, names=['conv_id', 'line'], sep='\t', header=None), pd.read_csv(training_predator_id_file, header=None, names=['tagged_pred']))
#         df_train.to_csv(f"{datapath}/toy.train/toy-train.csv")
#
#
#     if os.path.isfile(f'{datapath}toy.test/toy-test.csv'):
#         df_test = pd.read_csv(f'{datapath}toy.test/toy-test.csv', index_col=0).fillna('')
#     else:
#         test_file = f'{datapath}toy.test/pan12-sexual-predator-identification-test-corpus-2012-05-17.xml'
#         test_predator_id_file = f'{datapath}toy.test/pan12-sexual-predator-identification-groundtruth-problem1.txt'
#         test_tagged_msgs_file = f'{datapath}toy.test/pan12-sexual-predator-identification-groundtruth-problem2.txt'
#         df_test = read_xml(test_file, pd.read_csv(test_tagged_msgs_file, names=['conv_id', 'line'], sep='\t', header=None), pd.read_csv(test_predator_id_file, header=None, names=['tagged_pred']))
#         df_test.to_csv(f"{datapath}/toy.test/toy-test.csv")
#
#     # df_train_test = pd.concat([df_train, df_test])
#
# text_feature_sets = [["w2v_glove","nauthors", "time","count", "msg_line"]] # [["w2v_glove","prv_cat","nauthors", "time","count", "msg_line"]]
# Baselines = [msg_classifier()]#text_features, [len(df_train), len(df_test)], relabeling, df_train_test)]#, conv_msg_classifier(relabeling)]
#
# for text_feature_set in text_feature_sets:
#     text_feature_set_str = '.'.join(text_feature_set)
#     text_features = ef.extract_load_text_features(df_train_test, text_feature_set, f'../output/{text_feature_set_str}.npz')
#
#     for baseline in Baselines:
#         baseline.main(df_train_test, text_features, "../output/", text_feature_set_str)
#
# # 'tagged_msg': original labels (conv, msg_line) only available for test set
# # 'tagged_predator_bc': if conv has at least one predator, all the msgs of the conv are tagged
# # 'tagged_msg_bc': if conv has at least one tagged msg, all the msgs of the conv are tagged
# relabeling = ['tagged_msg', 'tagged_predator', 'tagged_conv']

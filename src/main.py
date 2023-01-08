from lxml import etree
import pandas as pd

import extract_features as ef
from classifier import msg_classifier
from classifier import conv_msg_classifier


def read_xml(xmlfile, tagged_msgs, predators):
    dictinary_list = []
    df = pd.DataFrame(columns=['conv_id', 'msg_id', 'author_id', 'time', 'text'], index=['conv_id'])
    root = etree.parse(xmlfile).getroot()  # <conversations>
    for conv in root.getchildren():
        for msg in conv.getchildren():
            author, time, body = msg.getchildren()
            row = {'conv_id': conv.get('id'),
                   'msg_line': msg.get('line'),
                   'author_id': author.text,
                   'time': time.text,
                   'text': '' if body.text is None else body.text,
                   'tagged_msg': 0 if tagged_msgs.loc[(tagged_msgs['conv_id'] == conv.get('id')) & (
                               tagged_msgs['line'] == int(msg.get('line')))].empty else 1,
                   'tagged_conv': 0 if tagged_msgs.loc[tagged_msgs['conv_id'] == conv.get('id')].empty else 1,
                   'tagged_predator': None if predators.empty else (1 if author.text in predators else 0),
                   }
            dictinary_list.append(row)
    return df.from_dict(dictinary_list)


def get_stats(conversations, predators, tagged_msgs):
    stats = {'n_convs': 0,
             'n_msgs': 0,
             'avg_n_msgs_convs': 0,
             'n_binconv': 0,
             'n_nonbinconv': 0,
             'avg_n_msgs_binconvs': 0,
             'avg_n_msgs_nonbinconvs': 0,

             'n_tagged_binconvs': 0,
             # needs relabeling: 1) any convs with at least one tagged_msg, 2) any convs with at least one predator
             'n_tagged_nonbinconvs': 0,  # needs relabeling
             'avg_n_msgs_tagged_convs': 0,

             'n_convs_mult_predators': 0,
             'avg_n_msgs_convs_for_predator': 0,
             'avg_n_normalconvs_for_predator': 0,
             }

    return stats


if __name__ == '__main__':
    datapath = '../data/toy.'

    training_file = f'{datapath}train/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml'
    training_predator_id_file = f'{datapath}train/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt'
    training_tagged_msgs_file = f'{datapath}train/pan12-sexual-predator-identification-diff.txt'

    test_file = f'{datapath}test/pan12-sexual-predator-identification-test-corpus-2012-05-17.xml'
    test_predator_id_file = f'{datapath}test/pan12-sexual-predator-identification-groundtruth-problem1.txt'
    test_tagged_msgs_file = f'{datapath}test/pan12-sexual-predator-identification-groundtruth-problem2.txt'

    df_train = read_xml(training_file, pd.read_csv(training_tagged_msgs_file, names=['conv_id', 'line'], sep='\t'),
                        pd.read_csv(training_predator_id_file))
    df_test = read_xml(test_file, pd.read_csv(test_tagged_msgs_file, names=['conv_id', 'line'], sep='\t'),
                       pd.read_csv(test_predator_id_file))

    df_train_test = pd.concat([df_train, df_test])

    text_feature_sets = [['w2v_glove']]  # [['basic'], ['w2v_glove'], ['w2v_bert']]
    for text_feature_set in text_feature_sets:
        text_feature_set_str = '.'.join(text_feature_set)
        text_features = ef.extract_load_text_features(df_train_test, text_feature_set,
                                                      f'../output/{text_feature_set_str}.npz')

    # 'tagged_msg': original labels (conv, msg_line) only available for test set
    # 'tagged_predator_bc': if conv has at least one predator, all the msgs of the conv are tagged
    # 'tagged_msg_bc': if conv has at least one tagged msg, all the msgs of the conv are tagged
    relabeling = ['tagged_msg', 'tagged_predator', 'tagged_conv']

    Baselines = [
        msg_classifier(text_features, [len(df_train), len(df_test)], relabeling)] # , conv_msg_classifier(relabeling)]

    for baseline in Baselines:
        baseline.main(df_train, df_test)

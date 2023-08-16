import os

import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import torchmetrics
from sklearn.metrics import auc
import nltk
from lxml import etree


def nltk_tokenize(input) -> list[list[str]]:
    tokens = [nltk.tokenize.word_tokenize(record.lower()) if pd.notna(record) else [] for record in input]
    return tokens

def force_open(path, *args, **kwargs):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    return open(path, *args, **kwargs)

def pan12_xml2csv(xmlfile, predatorsfile):
    with open(predatorsfile, "r") as f:
        predators = set(l.strip() for l in f.readlines())
        if len(predators) == 0:
            raise Exception(f"No predator was specified at '{predatorsfile}'")
    
    rows_list = []
    root = etree.parse(xmlfile).getroot()
    for counter, conv in enumerate(root.getchildren()):
        if counter % 500 == 0:
            print(counter)
        conversation_authors = set(conv.xpath("message/author/text()"))
        nauthor = len(conversation_authors)
        predatory_conversation = 1.0 if len(conversation_authors & predators) > 0 else 0.0
        for msg in conv.getchildren():
            author = msg.xpath("author/text()")
            author = author[0] if len(author) else ""

            time = msg.xpath("time/text()")
            time = time[0] if len(time) else ""

            body = msg.xpath("text/text()")
            body = body[0] if len(body) > 0 else ""

            row = [conv.get('id'), int(msg.get('line')), author, float(time.replace(":", ".")),
                   len(body) if body is not None else 0, len(body.split()) if body is not None else 0,
                   len(conv.getchildren()), nauthor, '' if body is None else body, 1.0 if author in predators else 0.0, predatory_conversation]
            rows_list.append(row)
    return pd.DataFrame(rows_list, columns=["conv_id", "msg_line", "author_id", "time", "msg_char_count", "msg_word_count", "conv_size", "nauthor", "text", "tagged_predator", "predatory_conv"])

def message_csv2conversation_csv(df):
    groups = df.sort_values(by=["conv_id", "msg_line"]).groupby("conv_id")
    conversations = []
    for _, group in groups:
        conversations.append((group["conv_id"].iloc[0], group["predatory_conv"].iloc[0], ". ".join(group["text"].fillna('')), group.shape[0], len(set(group["author_id"]))))
    
    return pd.DataFrame(conversations, columns=["conv_id", "predatory_conv", "text", "number_of_messages", "number_of_authors"])

def create_toy_dataset(df, fraction=0.1, keep_distribution=True):
    if keep_distribution:
        predatories = df[df["predatory_conv"] > 0.5].sample(frac=fraction)
        nonpredators = df[df["predatory_conv"] <= 0.5].sample(frac=fraction)
        new_df = pd.concat([predatories, nonpredators], axis=0)
        return new_df
    
    return df.sample(frac=fraction)

# ratio: predatory/(predatory+non-predatory)
def balance_dataset(dataset, ratio=0.5):
    predators_indices = dataset["predatory_conv"] == 1
    predators = dataset[predators_indices]
    non_predators = dataset[~predators_indices].sample(n=int(predators.shape[0] * (1-ratio)/ratio))

    return pd.concat([predators, non_predators], axis=0).sample(frac=1.0)

def get_stats_v2(data):
    predators_count = len(set(data[data["tagged_predator"] > 0.0]["author_id"]))
    chatters_count  = len(set(data["author_id"]))
    conversations = data.groupby("conv_id")
    predatory_conversations = data[data["predatory_conv"] > 0.0].groupby("conv_id")

    conversations_authors_count = conversations.apply(lambda x: len(set(x["author_id"])))
    conversations_messages_count = conversations.apply(lambda x: x.shape[0])

    predatory_conversations_authors_count = predatory_conversations.apply(lambda x: len(set(x["author_id"])))
    predatory_conversations_messages_count = predatory_conversations.apply(lambda x: x.shape[0])
    
    stats = {
        "number_of_chatters": chatters_count,
        "number_of_predatory_chatters": predators_count,
        "number_of_conversations": len(conversations),
        "number_of_messages": data.shape[0],
        "number_of_conversations_with_m_messages": {
            "m<=1": (conversations_messages_count <= 1).sum(),
            "m==2": (conversations_messages_count == 2).sum(),
            "m==3": (conversations_messages_count == 3).sum(),
            "m==4": (conversations_messages_count == 4).sum(),
            "m>=5": (conversations_messages_count >= 5).sum(),
        },
        "number_of_predatory_conversations_with_m_messages": {
            "m<=1": (predatory_conversations_messages_count <= 1).sum(),
            "m==2": (predatory_conversations_messages_count == 2).sum(),
            "m==3": (predatory_conversations_messages_count == 3).sum(),
            "m==4": (predatory_conversations_messages_count == 4).sum(),
            "m>=5": (predatory_conversations_messages_count >= 5).sum(),
        },
        "number_of_conversations_per_n_author": {
            "n==1":(conversations_authors_count == 1).sum(),
            "n==2": (conversations_authors_count == 2).sum(),
            "n>2":(conversations_authors_count > 2).sum(),
        },
        "average_number_of_conversations_messages_per_n_author": {
            "n==1": (conversations_messages_count[conversations_authors_count == 1].mean()),
            "n==2": (conversations_messages_count[conversations_authors_count == 2].mean()),
            "n>2": (conversations_messages_count[conversations_authors_count > 2].mean()),
        },
        
        "number_of_predatory_conversations": len(predatory_conversations),
        "number_of_predatory_conversations_per_n_author": {
            "n==1":(predatory_conversations_authors_count == 1).sum(),
            "n==2": (predatory_conversations_authors_count == 2).sum(),
            "n>2":(predatory_conversations_authors_count > 2).sum(),
        },
        "average_number_of_predatory_conversations_messages_per_n_author": {
            "n==1": (predatory_conversations_messages_count[predatory_conversations_authors_count == 1].mean()),
            "n==2": (predatory_conversations_messages_count[predatory_conversations_authors_count == 2].mean()),
            "n>2": (predatory_conversations_messages_count[predatory_conversations_authors_count > 2].mean()),
        },

        "average_conversations_per_predator": len(predatory_conversations) / predators_count,

    }
    
    return stats

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

def confusion_matrix(prediction, target, threshold=0.5):
    tp = ((prediction > threshold) & (target > threshold)).sum()
    fp = ((prediction > threshold) & ~(target > threshold)).sum()
    tn = (~(prediction > threshold) & ~(target > threshold)).sum()
    fn = (~(prediction > threshold) & (target > threshold)).sum()
    return tp, fp, tn, fn

def calculate_metrics(prediction, target, device="cpu"):
    accuracy = torchmetrics.Accuracy("binary").to(device)
    recall = torchmetrics.Recall("binary").to(device)
    precision = torchmetrics.Precision("binary").to(device)
    _p = prediction
    _t = target
    return accuracy(_p, _t), recall(_p, _t), precision(_p, _t)

def calculate_metrics_extended(prediction, target, device="cpu"):
    f2score = torchmetrics.FBetaScore("binary", beta=2.0).to(device)
    f05score = torchmetrics.FBetaScore("binary", beta=0.5).to(device)
    return *calculate_metrics(prediction, target, device), f2score(prediction, target), f05score(prediction, target)

def _calculate_metrics(prediction, target, *args, **kwargs):
    tp, fp, tn, fn = confusion_matrix(prediction, target)
    accuracy = (tp+tn) / (tp+tn+fp+fn)
    recall = tp / (tp+fn)
    precision = tp / (tp+fp)
    return accuracy, recall, precision

def roc(prediction, target, bins=None, device="cpu"):
    roc = torchmetrics.ROC("binary", thresholds=bins).to(device)    
    fpr, tpr, thresholds = roc(prediction, target.long())
    return fpr, tpr, thresholds

def roc_auc(prediction, target, bins=None, device="cpu"):
    auroc = torchmetrics.AUROC(task="binary", thresholds=bins).to(device)
    return auroc(prediction, target.long())

def precision_recall_curve(prediction, target, bins=None, device="cpu"):
    metric = torchmetrics.PrecisionRecallCurve("binary", thresholds=bins)
    if target.dtype not in (torch.long, torch.int):
        return metric(prediction, target.long())
    return metric(prediction, target)

def precision_recall_auc(prediction, target, bins=None, device="cpu"):
    precisions, recalls, _ = precision_recall_curve(prediction, target, bins, device)
    return auc(recalls.cpu(), precisions.cpu())

def _roc_auc(prediction, target, bins=100, *args, **kwargs):
    fprs, tprs = torch.zeros(bins, dtype=torch.float32), torch.zeros(bins, dtype=torch.float32)
    thresholds = np.linspace(0, 1, bins)
    for i, threshold in enumerate(thresholds):
        tp, fp, tn, fn = confusion_matrix(prediction, target, threshold=threshold)
        fpr = fp / (fp+tn)
        tpr = tp / (tp+fn)
        fprs[i] = fpr
        tprs[i] = tpr
    # It might be problematic some day but it works for now
    fprs[fprs.isnan()] = 0.0
    tprs[tprs.isnan()] = 0.0
    return fprs, tprs, thresholds


def padding_collate_sequence_batch(batch):
    
    label_list, data_list = [], []
    for data, label in batch:
        label_list.append(label)
        data_list.append(data.to_dense() if data.is_sparse else data)
    
    return pad_sequence(data_list, batch_first=True, padding_value=1.0/data_list[0].shape[-1]), torch.tensor(label_list)


class RegisterableObject:

    @classmethod
    def short_name(cls) -> str:
        raise NotImplementedError()


class CommandObject:

    @classmethod
    def command(cls) -> str:
        raise NotImplementedError()
    
    def get_actions_and_args(self):
        
        def action(*args, **kwargs):
            raise NotImplementedError("this command is not implemented fully")

        return (action, [dict()])
    
    def help(self) -> str:
        return f"just running {self.command()}"


class SingletonMeta(type):
    """
    You can make a class singleton using:
    class SingletonClass(metaclass=SingletonMeta):
        def do_something(self):
            pass
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

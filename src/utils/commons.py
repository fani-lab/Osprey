import os

import pandas as pd
from nltk.tokenize import word_tokenize
from lxml import etree


def nltk_tokenize(input) -> list[list[str]]:
    tokens = [word_tokenize(record.lower()) if pd.notna(record) else [] for record in input]
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

def message_csv2conversation_csv(path):
    df = pd.read_csv(path)
    groups = df.sort_values(by=["conv_id", "msg_line"])[["conv_id", "msg_line", "text", "predatory_conv"]].groupby("conv_id")
    conversations = []
    for name, group in groups:
        conversations.append((name, ". ".join(group["text"].fillna('')), group["predatory_conv"].iloc[0]))

    return pd.DataFrame(conversations, columns=["conv_id", "text", "predatory_conv"])

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


class RegisterableObject:

    @classmethod
    def short_name(cls) -> str:
        raise NotImplementedError()


import re

import pandas as pd

from src.utils.commons import message_csv2conversation_csv, force_open, balance_dataset, create_toy_dataset

def create_conversations():
    df = pd.read_csv("data/dataset-v2/train.csv")
    df = message_csv2conversation_csv(df)
    with force_open("data/dataset-v2/conversation/train.csv", mode="wb") as f:
        df.to_csv(f)
        del df
    
    df = pd.read_csv("data/dataset-v2/test.csv")
    df = message_csv2conversation_csv(df)
    with force_open("data/dataset-v2/conversation/test.csv", mode="wb") as f:
        df.to_csv(f)

def balance_datasets_for_version_two(ratio=0.3):
    train = "data/dataset-v2/conversation/train-v2.csv"
    test  = "data/dataset-v2/conversation/test-v2.csv"
    
    df = pd.read_csv(train)
    train = balance_dataset(df, ratio=ratio)
    train.to_csv(f"data/dataset-v2/conversation/balanced-train-v2-{str(ratio).replace('.', '')}.csv")

    df = pd.read_csv(test)
    test = balance_dataset(df, ratio=ratio)
    test.to_csv(f"data/dataset-v2/conversation/balanced-test-v2-{str(ratio).replace('.', '')}.csv")

def create_conversation_toy_set(train = "data/dataset-v2/conversation/balanced-train-v2-04.csv", test = "data/dataset-v2/conversation/balanced-test-v2-04.csv", ratio=0.1):
    df = pd.read_csv(train)
    df = create_toy_dataset(df, ratio)
    temp = re.split(r"(/|\\)", train)
    new_path = "".join(temp[:-1] + ["toy-" + temp[-1]])
    df.to_csv(new_path)
    
    temp = re.split(r"(/|\\)", test)
    new_path = "".join(temp[:-1] + ["toy-" + temp[-1]])
    df = pd.read_csv(test)
    df = create_toy_dataset(df, ratio)
    df.to_csv(new_path)

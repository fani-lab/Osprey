import pandas as pd

from src.utils.commons import message_csv2conversation_csv, force_open, balance_dataset

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

def balance_datasets_for_version_two():
    train = "data/dataset-v2/conversation/train-v2.csv"
    test  = "data/dataset-v2/conversation/test-v2.csv"
    
    df = pd.read_csv(train)
    train = balance_dataset(df, ratio=0.4)
    train.to_csv("data/dataset-v2/conversation/balanced-train-v2-04.csv")

    df = pd.read_csv(test)
    test = balance_dataset(df, ratio=0.4)
    test.to_csv("data/dataset-v2/conversation/balanced-test-v2-04.csv")


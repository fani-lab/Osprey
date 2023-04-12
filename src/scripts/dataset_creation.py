import pandas as pd

from src.utils.commons import message_csv2conversation_csv, force_open

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

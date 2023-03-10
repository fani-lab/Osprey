import os

import pandas as pd
from nltk.tokenize import word_tokenize

def nltk_tokenize(input) -> list[list[str]]:
    tokens = [word_tokenize(record.lower()) if pd.notna(record) else [] for record in input]
    return tokens

def force_open(path, *args, **kwargs):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    return open(path, *args, **kwargs)

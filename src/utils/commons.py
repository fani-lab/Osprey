import pandas as pd
from nltk.tokenize import word_tokenize

def nltk_tokenize(input) -> list[list[str]]:
    tokens = [word_tokenize(record.lower()) if pd.notna(record) else [] for record in input]
    return tokens
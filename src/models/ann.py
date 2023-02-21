from ast import literal_eval
import pickle

from baseline import Baseline
from preprocessing.base import BasePreprocessing
from utils.one_hot_encoder import GenerativeOneHotEncoder

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import nltk
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class SimpleANN(torch.nn.Module, Baseline):

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, preprocessings=list[BasePreprocessing], copy=True, load_from_pkl=True, preprocessed_path="data/preprocessed/basic/"):
        super(SimpleANN, self).__init__()
        self.preprocessings = preprocessings
        self.preprocessed_path = preprocessed_path
        self.load_from_pkl = load_from_pkl
        if copy:
            self.train_df = train.copy(deep=True)
            self.test_df  = test.copy(deep=True)
        else:
            self.train_df = train
            self.test_df  = test
        nltk.download('punkt')

    def _remove_stop_words(self):
        print("removing stopwords 1")
        stopwords_set = stopwords.words()
        print("removing stopwords 2")
        # train_tokens = []
        # for _, record in self.train_df.iterrows():
        #     train_tokens.append([token for token in record["tokens"] if token not in stopwords_set])
        self.train_df["tokens"] = self.train_df.apply(lambda row: [token for token in row["tokens"] if token not in stopwords_set], axis=1)
        self.test_df["tokens"]  = self.test_df.apply (lambda row: [token for token in row["tokens"] if token not in stopwords_set], axis=1)

    def vectorize_bow(self):
        print("generating one hot vectors")
        data = set()
        data.update(*[record.replace(",", "").replace("'", "")[1:-1].split() for record in self.train_df["tokens"]])
        data = list(data)
        one_hot_encoder = OneHotEncoder(handle_unknown='infrequent_if_exist')
        one_hot_encoder.fit([[record] for record in data])
        vectors = []
        for _, record in self.train_df.iterrows():
            try:
                temp = []
                for token in literal_eval(record["tokens"]):
                    if len(token) > 0:
                        temp.append([token])
                vectors.append(
                                one_hot_encoder.transform(temp).sum(axis=0) if len(temp) > 0 else np.zeros((1, len(data)), dtype=np.float64)
                                )
            except Exception as e:
                raise e
        return vectors

    def get_data_generator(self, data, pattern):
        def func():
            for record in data:
                yield pattern(record)
        
        return func

    def vectorize(self, tokens_records):
        data = set()
        data.update(*tokens_records)
        data = list(data)
        pattern = lambda x: x
        encoder = GenerativeOneHotEncoder(self.get_data_generator(data=data, pattern=pattern))
        encoder.fit()
        vectors = []
        for record in tokens_records:
            temp = encoder.transform(record=record)
            vectors.append(torch.sparse.sum(torch.cat(temp), dim=0))
        return vectors

    def nltk_tokenize(self, input) -> list[list[str]]:
        train_tokens = [word_tokenize(record.lower()) if pd.notna(record) else [] for record in input]
        return train_tokens

    def get_session_path(self, file_name: str = "") -> str:
        return self.preprocessed_path + file_name

    def prep(self):
        print("starting preperations")
        try:
            if not self.load_from_pkl:
                raise FileNotFoundError()
            with open(self.get_session_path("vectors.pkl"), "rb") as f:
                train_tokens = pickle.load(f)
            
        except FileNotFoundError:

            print("generating tokens")
            train_tokens = self.nltk_tokenize(self.train_df["text"])

            for preprocessor in self.preprocessings:
                train_tokens = [*preprocessor.opt(train_tokens)]
            train_tokens = self.vectorize(train_tokens)

            with open(self.preprocessed_path + "vectors.pkl", "wb") as f:
                pickle.dump(train_tokens, f)
        except Exception as e:
            raise e
        
        
        self.train_labels = np.array(self.train_df["tagged_msg"])
        self.train_tokens = train_tokens
        print("preparation is finished")

        
            


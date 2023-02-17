from ast import literal_eval

from baseline import Baseline
from preprocessing.stopwords import BasePreprocessing, NLTKStopWordRemoving

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class SimpleANN(Baseline):

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, preprocessings=list[BasePreprocessing], copy=True, load_from_pkl=True, preprocessed_path="data/preprocessed/basic/"):
        super(SimpleANN).__init__()
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
    
    def vectorize_bow2(self, tokens_records):
        print("generating one hot vectors")
        data = set()
        data.update(*tokens_records)
        data = list(data)
        one_hot_encoder = OneHotEncoder(handle_unknown='infrequent_if_exist')
        one_hot_encoder.fit([[record] for record in data])
        vectors = []
        for i, record in enumerate(tokens_records):
            if i == 76:
                print()
            temp = [[token] for token in record]
            vectors.append(
                        np.squeeze(np.asarray(one_hot_encoder.transform(temp).sum(axis=0))) if len(temp) > 0 else
                            np.zeros((len(data)), dtype=np.float64)
                    )
        return vectors

    def nltk_tokenize(self, input) -> list[list[str]]:
        train_tokens = [word_tokenize(record.lower()) if pd.notna(record) else [] for record in input]
        return train_tokens

    def prep(self):
        print("starting preperations")
        try:
            if not self.load_from_pkl:
                raise FileNotFoundError()
            self.train_df = pd.read_csv(self.preprocessed_path + "train.csv")
            train_tokens = np.array([np.fromstring(record[1:-1], dtype=float, sep=' ')
                                     for record in self.train_df["tokens"]])
        except FileNotFoundError:

            print("generating tokens")
            train_tokens = self.nltk_tokenize(self.train_df["text"])

            for preprocessor in self.preprocessings:
                train_tokens = [*preprocessor.opt(train_tokens)]
            train_tokens = self.vectorize_bow2(train_tokens)

            self.train_df["tokens"] = train_tokens

            self.train_df.to_csv(self.preprocessed_path + "train.csv", escapechar='\\')
            train_tokens = np.array(train_tokens)
        except Exception as e:
            raise e
        
        
        self.train_labels = np.array(self.train_df["tagged_msg"])
        self.train_tokens = train_tokens
        print("preparation is finished")
        
        
            


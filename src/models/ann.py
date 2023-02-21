from ast import literal_eval
import pickle
import logging

from baseline import Baseline
from preprocessing.base import BasePreprocessing
from utils.one_hot_encoder import GenerativeOneHotEncoder

import pandas as pd
import numpy as np
import nltk
import torch

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

logger = logging.getLogger()

class SimpleANN(torch.nn.Module, Baseline):

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, preprocessings=list[BasePreprocessing], copy=True, load_from_pkl=True, preprocessed_path="data/preprocessed/basic/", **kwargs):
        super(SimpleANN, self).__init__()
        logger.info("creating ann class")
        self.preprocessings = preprocessings
        self.preprocessed_path = preprocessed_path
        self.load_from_pkl = load_from_pkl
        
        self.encoder = GenerativeOneHotEncoder()

        if copy:
            logger.debug("copying df to ann class")
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
        self.train_df["tokens"] = self.train_df.apply(lambda row: [token for token in row["tokens"] if token not in stopwords_set], axis=1)
        self.test_df["tokens"]  = self.test_df.apply (lambda row: [token for token in row["tokens"] if token not in stopwords_set], axis=1)

    def get_data_generator(self, data, pattern):
        def func():
            for record in data:
                yield pattern(record)
        
        return func

    def vectorize(self, tokens_records):
        logger.info("started generating bag of words vectors")
        data = set()
        data.update(*tokens_records)
        data = list(data)
        pattern = lambda x: x
        logger.debug("fitting data into one hot encoder")
        self.encoder.fit(self.get_data_generator(data=data, pattern=pattern))
        logger.debug("started transforming message records into sparse vectors")
        vectors = []
        for record in tokens_records:
            temp = self.encoder.transform(record=record)
            vectors.append(torch.sparse.sum(torch.cat(temp), dim=0))
        logger.debug("transforming of records into vectors is finished")
        return vectors

    def nltk_tokenize(self, input) -> list[list[str]]:
        logger.debug("tokenizing using nltk")
        train_tokens = [word_tokenize(record.lower()) if pd.notna(record) else [] for record in input]
        logger.debug("finished toknizing using nltk")
        return train_tokens

    def get_session_path(self, file_name: str = "") -> str:
        return self.preprocessed_path + file_name

    def prep(self):
        logger.info("data preparation started")
        try:
            if not self.load_from_pkl:
                raise FileNotFoundError()
            logger.info("reading tokens and encoder from pickle files")
            logger.debug("reading vectors")
            with open(self.get_session_path("vectors.pkl"), "rb") as f:
                train_tokens = pickle.load(f)
            logger.debug("finished reading vectors")
            logger.debug("reading encoder")
            with open(self.get_session_path("one-hot-encoder.pkl"), "rb") as f:
                self.encoder = pickle.load(f)
            logger.debug("finished reading encoder")
            
        except FileNotFoundError:
            logger.info("generating tokens from scratch")
            train_tokens = self.nltk_tokenize(self.train_df["text"])
            logger.info("applying preprocessing modules")
            for preprocessor in self.preprocessings:
                logger.info(f"applying {preprocessor.name()}")
                train_tokens = [*preprocessor.opt(train_tokens)]
            logger.info("vectorizing data")
            train_tokens = self.vectorize(train_tokens)

            logger.info("saving vectors as pickle")
            with open(self.preprocessed_path + "vectors.pkl", "wb") as f:
                pickle.dump(train_tokens, f)

            logger.info("saving encoder as pickle")
            with open(self.get_session_path("one-hot-encoder.pkl"), "wb") as f: # TODO: save them via the vectorize method
                pickle.dump(self.encoder, f)
            
        except Exception as e:
            raise e
        
        
        self.train_labels = np.array(self.train_df["tagged_msg"])
        self.train_tokens = train_tokens
        logger.info("data preparation finished")

        
            


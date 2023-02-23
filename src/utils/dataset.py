import logging
import pickle

import pandas as pd
import torch
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize

from preprocessing.base import BasePreprocessing
from utils.one_hot_encoder import GenerativeOneHotEncoder
from utils.filing import force_open

logger = logging.getLogger()

class MyDataset(Dataset):
    def __init__(self, train, target):
        self.data = torch.stack(train)
        self.labels = torch.from_numpy(target)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)



class BagOfWordsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, output_path: str, load_from_pkl: bool, preprocessings=list[BasePreprocessing], parent_dataset=None, copy: bool=False):
        self.output_path = output_path
        self.parent_dataset = parent_dataset
        self.load_from_pkl = load_from_pkl
        self.preprocessings = preprocessings

        if copy:
            logger.debug("copying df to ann class")
            self.df = df.copy(deep=True)
        else:
            self.df = df

    def get_session_path(self, filename) -> str:
        return self.output_path + "bow/" + filename

    def prepare(self):
        logger.info("data preparation started")
        try:
            if not self.load_from_pkl:
                raise FileNotFoundError()
            logger.info("reading tokens and encoder from pickle files")
            logger.debug("reading vectors")
            with open(self.get_session_path("vectors.pkl"), "rb") as f:
                tokens = pickle.load(f)

            logger.debug("finished reading vectors")
            logger.debug("reading encoder")
            with open(self.get_session_path("one-hot-encoder.pkl"), "rb") as f:
                self.encoder = pickle.load(f)
            logger.debug("finished reading encoder")

        except FileNotFoundError:
            logger.info("generating tokens from scratch")
            tokens = self.nltk_tokenize(self.df["text"])
            logger.info("applying preprocessing modules")
            for preprocessor in self.preprocessings:
                logger.info(f"applying {preprocessor.name()}")
                tokens = [*preprocessor.opt(tokens)]
            logger.info("vectorizing data")
            encoder = self.init_encoder(tokens)
            tokens = self.vectorize(tokens, encoder)

            logger.info("saving vectors as pickle")
            with force_open(self.get_session_path("vectors.pkl"), "wb") as f:
                pickle.dump(tokens, f)

            logger.info("saving encoder as pickle")
            with force_open(self.get_session_path("one-hot-encoder.pkl"), "wb") as f: # TODO: save them via the vectorize method
                pickle.dump(self.encoder, f)

        except Exception as e:
            raise e

        self.labels = torch.tensor(self.df["tagged_msg"].values)
        self.data = torch.stack(tokens)
        logger.info("data preparation finished")
    
    def get_data_generator(self, data, pattern):
        def func():
            for record in data:
                yield pattern(record)

        return func
    
    def nltk_tokenize(self, input) -> list[list[str]]:
        logger.debug("tokenizing using nltk")
        tokens = [word_tokenize(record.lower()) if pd.notna(record) else [] for record in input]
        logger.debug("finished toknizing using nltk")
        return tokens
    
    def init_encoder(self, tokens_records):
        encoder = GenerativeOneHotEncoder()
        logger.info("started generating bag of words vector encoder")
        data = set()
        data.update(*tokens_records)
        pattern = lambda x: x
        logger.debug("fitting data into one hot encoder")
        encoder.fit(self.get_data_generator(data=data, pattern=pattern))

        return encoder
    
    def vectorize(self, tokens_records, encoder):
        logger.debug("started transforming message records into sparse vectors")
        vectors = []
        for record in tokens_records:
            temp = encoder.transform(record=record)
            vectors.append(torch.sparse.sum(torch.cat(temp), dim=0))
        logger.debug("transforming of records into vectors is finished")
        return vectors

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data.shape[0]

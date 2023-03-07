import logging
import pickle

import pandas as pd
import torch
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize

from preprocessing.base import BasePreprocessing
from utils.one_hot_encoder import GenerativeOneHotEncoder
from utils.filing import force_open
from imblearn.over_sampling import SMOTE

logger = logging.getLogger()


class MyDataset(Dataset):
    def __init__(self, train, target):
        if isinstance(train, pd.core.frame.DataFrame):
            self.data = torch.from_numpy(train.values)
            self.labels = torch.from_numpy(target.to_numpy())
        else:
            self.data = torch.stack(train)
            self.labels = torch.from_numpy(target)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

    @property
    def shape(self):
        return self.data.shape

    def oversample_by_smote(self):
        smote = SMOTE(random_state=42)
        self.data, self.labels = smote.fit_resample(self.data.to_dense(), self.labels)


class BagOfWordsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, output_path: str, load_from_pkl: bool,
                 preprocessings: list[BasePreprocessing] = [], persist_data=True, parent_dataset=None,
                 copy: bool = False):
        self.output_path = output_path
        self.parent_dataset = parent_dataset
        self.load_from_pkl = load_from_pkl
        self.preprocessings = preprocessings
        self.persist_data = persist_data

        if copy:
            logger.debug("copying df to ann class")
            self.df = df.copy(deep=True)
        else:
            self.df = df

    def get_session_path(self, filename) -> str:
        return self.output_path + "bow/" + filename

    def preprocess(self):
        try:
            if not self.load_from_pkl:
                raise FileNotFoundError()
            with open(self.get_session_path("tokens.pkl"), "rb") as f:
                tokens = pickle.load(f)
        except FileNotFoundError:
            logger.info("generating tokens from scratch")
            tokens = self.nltk_tokenize(self.df["text"])
            logger.info("applying preprocessing modules")
            for preprocessor in self.preprocessings:
                logger.info(f"applying {preprocessor.name()}")
                tokens = [*preprocessor.opt(tokens)]

        return tokens

    def prepare(self):
        tokens = self.preprocess()

        self.encoder = self.init_encoder(tokens_records=tokens)

        vectors = self.vectorize(tokens, self.encoder)

        # Persisting changes
        if self.persist_data:
            vectors_path = self.get_session_path("vectors.pkl")
            encoder_path = self.get_session_path("encoder.pkl")
            tokens_path = self.get_session_path("tokens.pkl")
            logger.info(f"saving tokens as pickle at {tokens_path}")
            with force_open(tokens_path, "wb") as f:
                pickle.dump(tokens, f)
            logger.info(f"saving vectors as pickle at {vectors_path}")
            with force_open(vectors_path, "wb") as f:
                pickle.dump(vectors, f)
            logger.info(f"saving encoder as pickle at {encoder_path}")
            with force_open(encoder_path, "wb") as f:
                pickle.dump(self.encoder, f)

        self.labels = torch.tensor(self.df["tagged_msg"].values)
        self.data = torch.stack(vectors)
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
        try:
            if self.parent_dataset is not None:
                encoder = self.parent_dataset.encoder
                return encoder
        except Exception as e:
            raise e

        try:
            if not self.load_from_pkl:
                raise FileNotFoundError()
            with open(self.get_session_path("one-hot-encoder.pkl"), "rb") as f:
                encoder = pickle.load(f)
        except FileNotFoundError:
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

    def oversample_by_smote(self):
        smote = SMOTE(random_state=42)
        self.data, self.labels = smote.fit_resample(self.data.to_dense(), self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data.shape[0]

    @property
    def shape(self):
        return self.data.shape



class TimeBasedBagOfWordsDataset(BagOfWordsDataset):

    def get_normalization_params(self, columns):
        if self.parent_dataset is not None:
            return self.parent_dataset.get_normalization_params(columns)
        if hasattr(self, "normalization_params"):
            return self.normalization_params

        self.normalization_params = dict()
        for c in columns:
            self.normalization_params[c] = (self.df[c].mean(), self.df[c].std())

        return self.normalization_params

    def vectorize(self, tokens_records, encoder):
        logger.debug("started transforming message records into sparse vectors")
        vectors = []
        context_columns = ("nauthor", "msg_line", "time")
        self.df[["nauthor", "msg_line", "time"]]
        normalization_params = self.get_normalization_params(context_columns)
        for c in context_columns:
            self.df[f"normalized_{c}"] = (self.df[c] - normalization_params[c][0]) / normalization_params[c][1]
        context_indices = [list(range(len(context_columns)))]

        for i, record in enumerate(tokens_records):
            onehots = torch.sparse.sum(torch.cat(encoder.transform(record=record)), dim=0)
            context = torch.sparse_coo_tensor(context_indices,
                                              [self.df.iloc[i][f"normalized_{c}"] for c in context_columns],
                                              (len(context_columns),), dtype=torch.float32)

            vectors.append(torch.hstack((context, onehots)))
        logger.debug("transforming of records into vectors is finished")
        return vectors


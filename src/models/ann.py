import pickle

import torchmetrics
from baseline import Baseline
from preprocessing.base import BasePreprocessing
from utils.one_hot_encoder import GenerativeOneHotEncoder

import pandas as pd
import numpy as np
import nltk
import torch
from torch import nn
from torch.utils.data import DataLoader
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from src.models.mydataset import MyDataset


class SimpleANN(torch.nn.Module, Baseline):
    #
    def __init__(self, dimension_list, activation, loss_func, lr, train: pd.DataFrame, test: pd.DataFrame,
                 preprocessings=list[BasePreprocessing], copy=True, load_from_pkl=True,
                 preprocessed_path="data/preprocessed/basic/"):

        # dimension_list = [950, 250, 150, 50, 2]
        super(SimpleANN, self).__init__()
        # Creating Model layers
        self.layers = nn.ModuleList()
        for i, j in zip(dimension_list, dimension_list[1:]):
            self.layers.append(nn.Linear(in_features=i, out_features=j))

        self.activation = activation
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        self.loss_function = loss_func

        self.preprocessings = preprocessings
        self.preprocessed_path = preprocessed_path
        self.load_from_pkl = load_from_pkl

        self.encoder = GenerativeOneHotEncoder()

        if copy:
            self.train_df = train.copy(deep=True)
            self.test_df = test.copy(deep=True)
        else:
            self.train_df = train
            self.test_df = test
        nltk.download('punkt')

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)

        x = self.layers[-1](x)
        return x

    def learn(self, epoch_num, batch_size):
        recall = torchmetrics.Recall(task='multiclass', num_classes=2)
        print(70 * '~')
        print('TRAINING PHASE')
        print(70 * '~')
        for i in range(1, epoch_num+1):
            loss = 0
            train_dataloader = DataLoader(self.train_dataset, batch_size, shuffle=True)
            for X, y in train_dataloader:
                y_hat = self.forward(X)
                self.optimizer.zero_grad()
                loss = self.loss_function(y_hat, y)
                loss.backward()
                self.optimizer.step()
                print(f'recall on batch: {recall(y_hat.argmax(1), y)}')

            print(f'epoch {i}:\n Loss: {loss}')

    def test(self):
        pass

    def _remove_stop_words(self):
        print("removing stopwords 1")
        stopwords_set = stopwords.words()
        print("removing stopwords 2")
        self.train_df["tokens"] = self.train_df.apply(
            lambda row: [token for token in row["tokens"] if token not in stopwords_set], axis=1)
        self.test_df["tokens"] = self.test_df.apply(
            lambda row: [token for token in row["tokens"] if token not in stopwords_set], axis=1)

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
        self.encoder.fit(self.get_data_generator(data=data, pattern=pattern))
        vectors = []
        for record in tokens_records:
            temp = self.encoder.transform(record=record)
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
        self.train_dataset = MyDataset(self.train_tokens, self.train_labels)
        print("preparation is finished")

import time
import logging
import sys

import pandas as pd
import torch.nn

from models.ann import SimpleANN
from preprocessing.stopwords import NLTKStopWordRemoving
from preprocessing.punctuations import PunctuationRemoving
from src.models.rnn import RnnModule
from utils.dataset import BagOfWordsDataset, TimeBasedBagOfWordsDataset

START_TIME = time.strftime("%m-%d-%Y-%H-%M-%S", time.localtime())

FORMATTER = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s : %(message)s")
FORMATTER_VERBOSE = logging.Formatter(
    "%(asctime)s | %(name)s | %(levelname)s | %(filename)s %(funcName)s @ %(lineno)s : %(message)s")

debug_file_handler = logging.FileHandler(f"logs/{START_TIME}.log")
debug_file_handler.setLevel(logging.DEBUG)
debug_file_handler.setFormatter(FORMATTER_VERBOSE)

info_terminal_handler = logging.StreamHandler(sys.stdout)
info_terminal_handler.setLevel(logging.INFO)
info_terminal_handler.setFormatter(FORMATTER)

logger = logging.getLogger()
logger.addHandler(debug_file_handler)
logger.addHandler(info_terminal_handler)
logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    test_path, train_path = "data/toy.test/toy-test.csv", "data/toy.train/toy-train.csv"
    logger.info("reading test csv file")
    test_df = pd.read_csv(test_path)
    logger.info("reading train csv file")
    train_df = pd.read_csv(train_path)
    logger.debug("reading test and train csv files is done")

    datasets_path = "/data/preprocessed/ann/1/"

    train_dataset = TimeBasedBagOfWordsDataset(train_df, datasets_path, True,
                                               preprocessings=[NLTKStopWordRemoving(), PunctuationRemoving()],
                                               copy=False)
    train_dataset.prepare()
    test_dataset = TimeBasedBagOfWordsDataset(test_df, datasets_path + "test_", True,
                                              preprocessings=[NLTKStopWordRemoving(), PunctuationRemoving()],
                                              parent_dataset=train_dataset, copy=False)
    test_dataset.prepare()
    ## data_size, hidden_size, output_size, activation, loss_func, lr, train: pd.DataFrame, test: pd.DataFrame, preprocessings = list[BasePreprocessing], copy = True, load_from_pkl = True, preprocessed_path = "data/preprocessed/basic/"
    kwargs = {
        "input_size": train_dataset.shape[1],
        "hidden_dim": 64,
        "num_layers": 1,
        "learning_batch_size": 64,
        "activation": torch.nn.ReLU(),
        "loss_func": torch.nn.CrossEntropyLoss(),
        "lr": 0.01,
        "train_dataset": train_dataset,
        "number_of_classes": 2,
    }
    model = RnnModule(**kwargs)

    model.learn(epoch_num=10, batch_size=64)
    model.test(test_dataset)
    print('Done!')

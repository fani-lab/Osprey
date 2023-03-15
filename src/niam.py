import time
import logging
import sys

import pandas as pd
import torch.nn

from src.models.ann import ANNModule
from src.preprocessing.stopwords import NLTKStopWordRemoving
from src.preprocessing.punctuations import PunctuationRemoving
from src.preprocessing.repetitions import RepetitionRemoving

from src.models.rnn import RnnModule
from src.utils.dataset import BagOfWordsDataset, TimeBasedBagOfWordsDataset, TransformersEmbeddingDataset

import settings

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

def main():
    # test_path, train_path = "data/toy.test/toy-test.csv", "data/toy.train/toy-train.csv"
    test_path, train_path = "data/test/test.csv", "data/train/train.csv"
    logger.info("reading test csv file")
    test_df = pd.read_csv(test_path)
    logger.info("reading train csv file")
    train_df = pd.read_csv(train_path)
    logger.debug("reading test and train csv files is done")

    datasets_path = "data/preprocessed/ann/"

    train_dataset = TimeBasedBagOfWordsDataset(train_path, datasets_path, False,
                                               preprocessings=[NLTKStopWordRemoving(), PunctuationRemoving(), RepetitionRemoving()])
    train_dataset.prepare()
    test_dataset = TimeBasedBagOfWordsDataset(test_path, datasets_path + "test-", False,
                                              preprocessings=[NLTKStopWordRemoving(), PunctuationRemoving(), RepetitionRemoving()],
                                              parent_dataset=train_dataset)
    test_dataset.prepare()
    ## data_size, hidden_size, output_size, activation, loss_func, lr, train: pd.DataFrame, test: pd.DataFrame, preprocessings = list[BasePreprocessing], copy = True, load_from_pkl = True, preprocessed_path = "data/preprocessed/basic/"
    # kwargs = {
    #     "input_size": train_dataset.shape[1],
    #     "hidden_dim": 64,
    #     "num_layers": 1,
    #     "learning_batch_size": 64,
    #     "activation": torch.nn.ReLU(),
    #     "loss_func": torch.nn.CrossEntropyLoss(),
    #     "lr": 0.001,
    #     "train_dataset": train_dataset,
    #     "module_session_path": "/output/rnn",
    #     "number_of_classes": 2,
    # }


    kwargs = {
        # "dimension_list": list([950, 250, 150, 50, 2]),
        "dimension_list": list([128]),
        "activation": torch.nn.ReLU(),
        "loss_func": torch.nn.CrossEntropyLoss(),
        "lr": 0.1,
        "train_dataset": train_dataset,
        "module_session_path": f"output/{START_TIME}/",
        "number_of_classes": 2,
    }
    # model = ANNModule(**kwargs)

    # model.learn(epoch_num=100, batch_size=64, k_fold=10)
    # model.test(test_dataset)
    logger.info('Done!')


def run():
    datasets = dict()
    for dataset_name, (short_name, train, test) in settings.datasets.items():
        dataset_class = None
        if short_name == "bow":
            dataset_class = BagOfWordsDataset
        elif short_name == "transformer/":
            dataset_class = TransformersEmbeddingDataset
        else:
            raise Exception(f"the dataset {short_name} is not implemented.")
        train_dataset = dataset_class(**train)
        datasets[dataset_name] = (train_dataset, dataset_class(**test, parent_dataset=train_dataset))

    for model_name, session in settings.sessions.items():
        commands = session["commands"]
        session["folds_number"]
        session["dataset"]
        model_configs = session["model_configs"]
        model_class = None
        
        if model_name == "ann":
            model_class = ANNModule
        
        for command, command_kwargs, dataset_name, *_ in commands:
            model = model_class(**model_configs, input_size=datasets[dataset_name][0].shape[1])

            if command == "train":
                model.learn(**command_kwargs, train_dataset=datasets[dataset_name][0])
            if command == "test":
                model.test(**command_kwargs, test_dataset=datasets[dataset_name][1])
            if command == "evaluate":
                model.eval(**command_kwargs)

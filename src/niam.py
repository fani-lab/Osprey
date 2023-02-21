import time
import logging
import sys

import pandas as pd
from models.ann import SimpleANN
from preprocessing.stopwords import NLTKStopWordRemoving
from preprocessing.punctuations import PunctuationRemoving


START_TIME = time.strftime("%m-%d-%Y-%H-%M-%S", time.localtime() )

FORMATTER = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s : %(message)s")
FORMATTER_VERBOSE = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(filename)s %(funcName)s @ %(lineno)s : %(message)s")

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
    test_path, train_path = "data/test/test.csv", "data/train/train.csv"
    test_path, train_path = "data/toy.test/toy-test.csv", "data/toy.train/toy-train.csv"
    kwargs = {
        # "preprocessed_path": "data/preprocessed/basic/",
        "preprocessed_path": "data/preprocessed/basic/generative/4toy-",
        "load_from_pkl": False,
        "preprocessings": [NLTKStopWordRemoving(), PunctuationRemoving()],
    }
    logger.info("reading test csv file")
    test_df  = pd.read_csv(test_path)
    logger.info("reading train csv file")
    train_df = pd.read_csv(train_path)
    logger.debug("reading test and train csv files is done")

    model = SimpleANN(train_df, test_df, **kwargs)
    
    try:
        model.prep()
    except Exception as e:
        logger.error(e)
        raise e

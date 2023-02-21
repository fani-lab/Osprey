import pandas as pd
import torch.nn

from models.ann import SimpleANN
from preprocessing.stopwords import NLTKStopWordRemoving
from preprocessing.punctuations import PunctuationRemoving

if __name__ == "__main__":
    # test_path, train_path = "data/test/test.csv", "data/train/train.csv"
    test_path, train_path = "../data/toy.test/toy-test.csv", "../data/toy.train/toy-train.csv"
    test_df = pd.read_csv(test_path)
    train_df = pd.read_csv(train_path)

    # dimension_list, activation, loss_func, lr, train: pd.DataFrame, test: pd.DataFrame, preprocessings = list[BasePreprocessing], copy = True, load_from_pkl = True, preprocessed_path = "data/preprocessed/basic/"
    kwargs = {
        "dimension_list": list([950, 250, 150, 50, 2]),
        "activation": torch.nn.ReLU(),
        "loss_func": torch.nn.CrossEntropyLoss(),
        "lr": 0.001,
        "train": train_df,
        "test": test_df,
        # "preprocessed_path": "data/preprocessed/basic/",
        "preprocessed_path": "../data/preprocessed/basic/toy",
        "load_from_pkl": True,
        "preprocessings": [NLTKStopWordRemoving(), PunctuationRemoving()],
    }
    model = SimpleANN(**kwargs)

    try:
        model.prep()
    except Exception as e:
        # e.with_traceback()
        raise e

    model.learn(epoch_num=10, batch_size=64)

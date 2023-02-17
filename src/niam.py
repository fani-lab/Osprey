import pandas as pd
from models.ann import SimpleANN
from preprocessing.stopwords import NLTKStopWordRemoving, PunctuationRemoving

if __name__ == "__main__":
    test_path, train_path = "data/test/test.csv", "data/train/train.csv"
    test_path, train_path = "data/toy.test/toy-test.csv", "data/toy.train/toy-train.csv"
    kwargs = {
        # "preprocessed_path": "data/preprocessed/basic/",
        "preprocessed_path": "data/preprocessed/basic/toyy-",
        "load_from_pkl": False,
        "preprocessings": [NLTKStopWordRemoving(), PunctuationRemoving()],
    }
    test_df  = pd.read_csv(test_path)
    train_df = pd.read_csv(train_path)

    model = SimpleANN(train_df, test_df, **kwargs)
    
    try:
        model.prep()
    except Exception as e:
        # e.with_traceback()
        raise e

TRAIN = 1
TEST  = 2
EVAL  = 4


preconfiged_sessions = {
    "sample":{
        "commands": [
            ("train", {"epochs": 100, "batch_size": 64, "folds": 10}, "train_bow"),
            ("test", {}, "test_bow")],
        "dataset": "contextBoW",
        
        "model_configs": {
            # Custom configs of a model as dict
        },
    },
    "ann": {
        "commands": ["train", "test", "eval"],
        "folds_number": 3,
        "dataset": "",
        "model_configs": {
            "dimension_list": list([128]),
            "activation": "relu",
            "loss_func": "cross-entropy",
            "lr": 0.1,
            "module_session_path": f"output/",
            "session_path_include_time": True,
            "number_of_classes": 2,
        },
    },
}

preconfiged_datasets = {
    "bow-v0": (
        "bow",  # short name of the dataset
        {       # train configs
            "data_path": "data/train/train.csv",
            "output_path": "data/preprocessed/ann/",
            "load_from_pkl": True,
            "preprocessings": [],
            "persist_data": True,
        },
        {      # test configs
            "data_path": "data/test/test.csv",
            "output_path": "data/preprocessed/ann/",
            "load_from_pkl": True,
            "preprocessings": [],
            "persist_data": True,
        }
    ),
}

sessions = {
    "ann": preconfiged_sessions["ann"],
}

datasets = preconfiged_datasets

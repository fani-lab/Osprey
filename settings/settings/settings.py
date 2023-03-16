TRAIN = 1
TEST  = 2
EVAL  = 4


preconfiged_sessions = {
    "sample":{
        "commands": [
            ("train", {"epoch_num": 100, "batch_size": 64, "k_fold": 10}, "bow-v0"),
            ("test", {}, "test_bow")],
        
        "model_configs": {
            # Custom configs of a model as dict
        },
    },
    "ann": {
        "commands": [
            ("train", {"epoch_num": 110, "batch_size": 500, "k_fold": 10}, "bow-v0"),
            ("test", dict(), "bow-v0"),
            ("eval", {"path": 'output/ann-test/ann/'}, ""),
            ],
        "model_configs": {
            "dimension_list": list([32]),
            "activation": ("relu", dict()),
            "loss_func": ("BCEW", dict()),
            "lr": 0.01,
            "module_session_path": "output/ann-test/",
            "session_path_include_time": False,
            "number_of_classes": 1,
            "device": 'cuda'
        },
    },
}

preconfiged_datasets = {
    "bow-v0": (
        "bow",  # short name of the dataset
        {       # train configs
            "data_path": "data/toy.train/toy-train.csv",
            "output_path": "data/preprocessed/ann/",
            "load_from_pkl": True,
            "preprocessings": ["sw", "pr", "rr"],
            "persist_data": True,
        },
        {      # test configs
            "data_path": "data/toy.test/toy-test.csv",
            "output_path": "data/preprocessed/ann/test-",
            "load_from_pkl": True,
            "preprocessings": ["sw", "pr", "rr"],
            "persist_data": True,
        }
    ),
}

sessions = {
    "ann": preconfiged_sessions["ann"],
}

datasets = preconfiged_datasets

FILTERED_CONFIGS = {
    "session_path_include_time",
    "data_path",
    "output_path",
    "load_from_pkl",
    "preprocessings",
    "persist_data",
}
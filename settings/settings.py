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
            ("train", {"epoch_num": 100, "batch_size": 64, "k_fold": 10}, "bow-v0"),
             ("test", dict(), "bow-v0"),
             ("eval", dict(), ""),
            ],
        "model_configs": {
            "dimension_list": list([128]),
            "activation": ("relu", dict()),
            "loss_func": ("cross-entropy", dict()),
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
            "data_path": "data/toy.train/toy-train.csv",
            "output_path": "data/preprocessed/ann/",
            "load_from_pkl": True,
            "preprocessings": ["sw", "pr", "rr"],
            "persist_data": True,
        },
        {      # test configs
            "data_path": "data/toy.test/toy-test.csv",
            "output_path": "data/preprocessed/ann/",
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

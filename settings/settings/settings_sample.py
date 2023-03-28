datasets = {
    "bow-onehot": (
        "bow",  # short name of the dataset
        {       # train configs
            "data_path": "data/balanced/train.csv",
            "output_path": "data/preprocessed/balanced/ann/",
            "load_from_pkl": True,
            "preprocessings": ["sw", "pr", "rr"],
            "persist_data": True,
        },
        {      # test configs
            "data_path": "data/balanced/train.csv",
            "output_path": "data/preprocessed/balanced/ann/test-",
            "load_from_pkl": True,
            "preprocessings": ["sw", "pr", "rr"],
            "persist_data": True,
        }
    ),

    "glove-v0": (
        "glove/twitter.50d",  # short name of the dataset
        {       # train configs
            "data_path": "data/balanced/train.csv",
            "output_path": "data/preprocessed/lolbalanced-ann/",
            "load_from_pkl": True,
            "preprocessings": ["sw"],
            "persist_data": False,
        },
        {      # test configs
            "data_path": "data/balanced/test.csv",
            "output_path": "data/preprocessed/lolbalanced-ann/test-",
            "load_from_pkl": True,
            "preprocessings": ["sw"],
            "persist_data": False,
        }
    ),
}

sessions = {
    "ann-onehot": {
        "model": "ann",
        "commands": [
            # ("train", {"epoch_num": 2, "batch_size": 64, "k_fold": 3}, "bow-onehot"),
            ("test", dict(), "bow-onehot"),
            ("eval", {"path": 'output/ann/', "use_current_session": True}, None),
            ],
        "model_configs": {
            "dimension_list": list([64, 16]),
            "activation": ("relu", dict()),
            "loss_func": ("BCEW", dict()),
            "lr": 0.9,
            "module_session_path": "output",
            "session_path_include_time": False,
            # "number_of_classes": 2,
            # "device": 'cuda'
        },
    },
}

USE_CUDA_IF_AVAILABLE = True

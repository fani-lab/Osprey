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
            # ("eval", {"path": 'output/ann/'}, ""),
            ],
        "model_configs": {
            "dimension_list": list([32]),
            "activation": ("relu", dict()),
            "loss_func": ("BCEW", dict()),
            "lr": 0.01,
            "module_session_path": "output/ann/",
            "session_path_include_time": False,
            "number_of_classes": 1,
            "device": 'cuda'
        },
    },
}

preconfiged_datasets = {
    # "bow-v0": (
    #     "bow",  # short name of the dataset
    #     {       # train configs
    #         "data_path": "data/toy.train/toy-train.csv",
    #         "output_path": "data/preprocessed/ann/",
    #         "load_from_pkl": True,
    #         "preprocessings": ["sw", "pr", "rr"],
    #         "persist_data": True,
    #     },
    #     {      # test configs
    #         "data_path": "data/toy.test/toy-test.csv",
    #         "output_path": "data/preprocessed/ann/test-",
    #         "load_from_pkl": True,
    #         "preprocessings": ["sw", "pr", "rr"],
    #         "persist_data": True,
    #     }
    # ),

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

    # "balanced-bert-base-cased": (
    #     "tranformer/bert-base-cased",  # short name of the dataset
    #     {       # train configs
    #         "data_path": "data/balanced/train.csv",
    #         "output_path": "data/preprocessed/balanced-ann/",
    #         "load_from_pkl": True,
    #         "preprocessings": ["sw", "pr", "rr"],
    #         "persist_data": True,
    #     },
    #     {      # test configs
    #         "data_path": "data/balanced/test.csv",
    #         "output_path": "data/preprocessed/balanced-ann/test-",
    #         "load_from_pkl": True,
    #         "preprocessings": ["sw", "pr", "rr"],
    #         "persist_data": True,
    #     }
    # ),
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

    # "ann-word2vec": {
    #     "commands": [
    #         ("train", {"epoch_num": 110, "batch_size": 500, "k_fold": 10}, "bow-v0"),
    #         ("test", dict(), "bow-v0"),
    #         ("eval", {"path": 'output/ann/'}, ""),
    #         ],
    #     "model_configs": {
    #         "dimension_list": list([32]),
    #         "activation": ("relu", dict()),
    #         "loss_func": ("BCEW", dict()),
    #         "lr": 0.01,
    #         "module_session_path": "output",
    #         "session_path_include_time": False,
    #         "number_of_classes": 1,
    #         "device": 'cuda'
    #     },
    # },

    # "ann-glove": {
    #     "model": "ann",
    #     "commands": [
    #         ("train", {"epoch_num": 3, "batch_size": 128, "k_fold": 2}, "glove-v0"),
    #         ("test", dict(), "glove-v0"),
    #         # ("eval", {"path": 'output/ann/'}, ""),
    #         ],
    #     "model_configs": {
    #         "dimension_list": list([32, 8]),
    #         "activation": ("relu", dict()),
    #         "loss_func": ("BCEW", {"reduction": "sum"}),
    #         "lr": 0.001,
    #         "module_session_path": "output",
    #         "session_path_include_time": False,
    #         "number_of_classes": 1,
    #         "device": 'cuda'
    #     },
    # },
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

IGNORED_PARAM_RESET = {"activation", "loss_function"}

USE_CUDA_IF_AVAILABLE = True

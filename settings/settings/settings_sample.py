import torch


datasets = {
    "toy-conversation-v2-dataset-onehot-raw": (
        "conversation-bow",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/conversation/toy-balanced-train-v2-04.csv",
            "output_path": "data/preprocessed/toy-conversation-v2/",
            "load_from_pkl": True,
            "preprocessings": ["rr"],
            "persist_data": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/conversation/toy-balanced-test-v2-04.csv",
            "output_path": "data/preprocessed/toy-conversation-v2/test-",
            "load_from_pkl": True,
            "preprocessings": ["rr"],
            "persist_data": True,
            "vector_size": 7500, # default is -1 which will be infered by the model
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

    "conversation-with-index-v2-dataset-onehot-raw-balanced-04": (
        "conversation-bow-with-triple",
        {       # train configs
            "data_path": "data/dataset-v2/conversation/balanced-train-v2-04.csv", # for getting the datasets, contact the corresponding author
            "output_path": "data/preprocessed/conversation-balanced-v2-04/",
            "load_from_pkl": True,
            "preprocessings": ["rr", "idr"],
            "persist_data": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/conversation/balanced-test-v2-04.csv",
            "output_path": "data/preprocessed/conversation-balanced-v2-04/test-",
            "load_from_pkl": True,
            "preprocessings": ["rr", "idr"],
            "persist_data": True,
        }
    ),

    "sequential-conversation-v2-dataset-onehot": (
        "basic-sequential",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/train-04.csv",
            "output_path": "data/preprocessed/sequential-v2-04/",
            "load_from_pkl": True,
            "preprocessings": ["rr", "idr"],
            "persist_data": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test-04.csv",
            "output_path": "data/preprocessed/sequential-v2-04/test-",
            "load_from_pkl": True,
            "preprocessings": ["rr", "idr"],
            "persist_data": True,
        }
    ),
}

sessions = {
    "balanced-v2-04": { # you can use this sample for running the dynamic superloss model
        "model": "ann-with-superloss",
        "commands": [
            ("train", {
                    "epoch_num": 75,
                    "batch_size": 8,
                    "weights_checkpoint_path": "",
                },
                {
                    "dataset": "conversation-v2-dataset-onehot-raw-balanced-04",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": None,
                    "n_splits": 4,
                }
            ),
            ("test", dict(), {"dataset": "conversation-v2-dataset-onehot-raw-balanced-04"}),
            ("eval", {"path": '', "use_current_session": True}, dict()),
        ],
        "model_configs": {
            "dimension_list": list([32]),
            "dropout_list": [0.0],
            "activation": ("relu", dict()),
            # "loss_func": ("weighted-binary-cross-entropy", {"pos_weight": 1.8}),
            "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(0.75)}),
            "lr": 0.005,
            "module_session_path": "output",
            "session_path_include_time": True,
        },
    },

    "lstm-balanced-v2-04": {
        "model": "lstm",
        "commands": [
            ("train", {
                    "epoch_num": 100,
                    "batch_size": 16,
                    "weights_checkpoint_path": "",
                },
                {
                    "dataset": "sequential-conversation-v2-dataset-onehot",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": None,
                    "n_splits": 5,
                }
            ),
            ("test", dict(), {"dataset": "sequential-conversation-v2-dataset-onehot"}),
            ("eval", {"path": '', "use_current_session": True}, dict()),
        ],
        "model_configs": {
            "activation": ("relu", dict()),
            "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(1.2)}),
            "lr": 0.0005,
            'hidden_size': 512,
            'num_layers': 1,
            "module_session_path": "output",
            "session_path_include_time": True,
        },
    },

    "toy-balanced-v2-04": {
        "model": "ann",
        "commands": [
            ("train", {
                    "epoch_num": 100,
                    "batch_size": 8,
                    "weights_checkpoint_path": "",
                },
                {
                    "dataset": "toy-conversation-v2-dataset-onehot-raw",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": None,
                    "n_splits": 3,
                }
            ),
            ("test", dict(), {"dataset": "toy-conversation-v2-dataset-onehot-raw"}),
            ("eval", {"path": '', "use_current_session": True}, dict()),
        ],
        "model_configs": {
            "dimension_list": list([32]),
            "dropout_list": [0.0],
            "activation": ("relu", dict()),
            "loss_func": ("weighted-binary-cross-entropy", {"pos_weight": 1.5}),
            # "loss_func": ("weighted-binary-cross-entropy", {"reduction": "sum"}),
            "lr": 0.0008,
            "module_session_path": "output",
            "session_path_include_time": False,
        },
    },
}

USE_CUDA_IF_AVAILABLE = True

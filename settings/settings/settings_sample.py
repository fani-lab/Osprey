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

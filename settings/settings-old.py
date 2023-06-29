import torch

TRAIN = 1
TEST  = 2
EVAL  = 4


datasets = {
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

    # "bow-onehot": (
    #     "bow",  # short name of the dataset
    #     {       # train configs
    #         "data_path": "data/balanced/train.csv",
    #         "output_path": "data/preprocessed/balanced/ann/",
    #         "load_from_pkl": True,
    #         "preprocessings": ["sw", "pr", "rr"],
    #         "persist_data": True,
    #     },
    #     {      # test configs
    #         "data_path": "data/balanced/train.csv",
    #         "output_path": "data/preprocessed/balanced/ann/test-",
    #         "load_from_pkl": True,
    #         "preprocessings": ["sw", "pr", "rr"],
    #         "persist_data": True,
    #     }
    # ),

    # "glove-v0": (
    #     "glove/twitter.50d",  # short name of the dataset
    #     {       # train configs
    #         "data_path": "data/balanced/train.csv",
    #         "output_path": "data/preprocessed/lolbalanced-ann/",
    #         "load_from_pkl": True,
    #         "preprocessings": ["sw"],
    #         "persist_data": False,
    #     },
    #     {      # test configs
    #         "data_path": "data/balanced/test.csv",
    #         "output_path": "data/preprocessed/lolbalanced-ann/test-",
    #         "load_from_pkl": True,
    #         "preprocessings": ["sw"],
    #         "persist_data": False,
    #     }
    # ),

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

    "v2-dataset-onehot": (
        "bow",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/train.csv",
            "output_path": "data/preprocessed/dataset-v2/",
            "load_from_pkl": True,
            "preprocessings": [],
            "persist_data": False,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "output_path": "data/preprocessed/dataset-v2/test-",
            "load_from_pkl": True,
            "preprocessings": [],
            "persist_data": False,
        }
    ),

    "conversation-v2-dataset-onehot": (
        "conversation-bow",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/conversation/train.csv",
            "output_path": "data/preprocessed/conversation-dataset-v2/",
            "load_from_pkl": True,
            "preprocessings": ["rr"],
            "persist_data": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/conversation/test.csv",
            "output_path": "data/preprocessed/conversation-dataset-v2/test-",
            "load_from_pkl": True,
            "preprocessings": ["rr"],
            "persist_data": True,
        }
    ),
    "toy-conversation-v2-dataset-onehot": (
        "conversation-bow",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/conversation/toy-train.csv",
            "output_path": "data/preprocessed/conversation-dataset-v2/toy-",
            "load_from_pkl": True,
            "preprocessings": ["pr", "sw", "rr"],
            "persist_data": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/conversation/toy-test.csv",
            "output_path": "data/preprocessed/conversation-dataset-v2/toy-test-",
            "load_from_pkl": True,
            "preprocessings": ["pr", "sw", "rr"],
            "persist_data": True,
        }
    ),

    "cnn-conversation-v2-dataset-onehot": (
        "cnn-conversation-bow",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/conversation/train.csv",
            "output_path": "data/preprocessed/conversation-dataset-v2/",
            "load_from_pkl": True,
            "preprocessings": ["rr"],
            "persist_data": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/conversation/test.csv",
            "output_path": "data/preprocessed/conversation-dataset-v2/test-",
            "load_from_pkl": True,
            "preprocessings": ["rr"],
            "persist_data": True,
        }
    ),

    "conversation-v2-dataset-onehot-balanced-04": (
        "conversation-bow-cleaned",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/conversation/balanced-train-v2-04.csv",
            "output_path": "data/preprocessed/conversation-balanced-test-v2-04/",
            "load_from_pkl": True,
            "preprocessings": ["rr"],
            "persist_data": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/conversation/balanced-test-v2-04.csv",
            "output_path": "data/preprocessed/conversation-balanced-test-v2-04/test-",
            "load_from_pkl": True,
            "preprocessings": ["rr"],
            "persist_data": True,
        }
    ),
################################
    "conversation-v2-dataset-onehot-raw": (
        "conversation-bow",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/conversation/train-v2.csv",
            "output_path": "data/preprocessed/conversation-v2/",
            "load_from_pkl": True,
            "preprocessings": ["rr"],
            "persist_data": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/conversation/test-v2.csv",
            "output_path": "data/preprocessed/conversation-v2/test-",
            "load_from_pkl": True,
            "preprocessings": ["rr"],
            "persist_data": True,
        }
    ),
    "conversation-v2-dataset-onehot-raw-balanced-04": (
        "conversation-bow",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/conversation/balanced-train-v2-04.csv",
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

    "conversation-v2-dataset-onehot-raw-balanced-03": (
        "conversation-bow",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/conversation/balanced-train-v2-03.csv",
            "output_path": "data/preprocessed/conversation-balanced-v2-03/",
            "load_from_pkl": True,
            "preprocessings": ["rr"],
            "persist_data": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/conversation/balanced-test-v2-03.csv",
            "output_path": "data/preprocessed/conversation-balanced-v2-03/test-",
            "load_from_pkl": True,
            "preprocessings": ["rr"],
            "persist_data": True,
        }
    ),

    "conversation-v2-dataset-onehot-raw-balanced-02": (
        "conversation-bow",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/conversation/balanced-train-v2-02.csv",
            "output_path": "data/preprocessed/conversation-balanced-v2-02/",
            "load_from_pkl": True,
            "preprocessings": ["rr"],
            "persist_data": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/conversation/balanced-test-v2-02.csv",
            "output_path": "data/preprocessed/conversation-balanced-v2-02/test-",
            "load_from_pkl": True,
            "preprocessings": ["rr"],
            "persist_data": True,
        }
    ),

    "conversation-v2-dataset-onehot-raw-balanced-01": (
        "conversation-bow",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/conversation/balanced-train-v2-01.csv",
            "output_path": "data/preprocessed/conversation-balanced-v2-01/",
            "load_from_pkl": True,
            "preprocessings": ["rr"],
            "persist_data": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/conversation/balanced-test-v2-01.csv",
            "output_path": "data/preprocessed/conversation-balanced-v2-01/test-",
            "load_from_pkl": True,
            "preprocessings": ["rr"],
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

    # Temporal Sequentials of different imbalance ratio
    ################################################
    "temporal-sequential-conversation-v2-dataset-onehot-04": (
        "temporal-sequential",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/train-04.csv",
            "output_path": "data/preprocessed/sequential-v2-04/",
            "load_from_pkl": True,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": True,
            "vector_size": 13000,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test-04.csv",
            "output_path": "data/preprocessed/sequential-v2-04/test-",
            "load_from_pkl": True,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": True,
            "vector_size": 13000,
        }
    ),

    "temporal-sequential-conversation-v2-dataset-onehot-03": (
        "temporal-sequential",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/train-03.csv",
            "output_path": "data/preprocessed/sequential-v2-03/",
            "load_from_pkl": True,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": True,
            "vector_size": 13000,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test-03.csv",
            "output_path": "data/preprocessed/sequential-v2-03/test-",
            "load_from_pkl": True,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": True,
            "vector_size": 13000,
        }
    ),

    "temporal-sequential-conversation-v2-dataset-onehot-02": (
        "temporal-sequential",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/train-02.csv",
            "output_path": "data/preprocessed/sequential-v2-02/",
            "load_from_pkl": True,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": True,
            "vector_size": 11000,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test-02.csv",
            "output_path": "data/preprocessed/sequential-v2-02/test-",
            "load_from_pkl": True,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": True,
            "vector_size": 11000,
        }
    ),

    "temporal-sequential-conversation-v2-dataset-onehot-01": (
        "temporal-sequential",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/train-01.csv",
            "output_path": "data/preprocessed/sequential-v2-01/",
            "load_from_pkl": True,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": True,
            "vector_size": 13000,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test-01.csv",
            "output_path": "data/preprocessed/sequential-v2-01/test-",
            "load_from_pkl": True,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": True,
            "vector_size": 13000,
        }
    ),

    # sequentials of different imbalance ratio
    ##################
    "sequential-conversation-v2-dataset-onehot-03": (
        "basic-sequential",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/train-03.csv",
            "output_path": "data/preprocessed/sequential-v2-03/",
            "load_from_pkl": True,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": True,
            "vector_size": 13000,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test-03.csv",
            "output_path": "data/preprocessed/sequential-v2-03/test-",
            "load_from_pkl": True,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": True,
            "vector_size": 13000,
        }
    ),

    "sequential-conversation-v2-dataset-onehot-02": (
        "basic-sequential",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/train-02.csv",
            "output_path": "data/preprocessed/sequential-v2-02/",
            "load_from_pkl": False,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": True,
            "vector_size": 7500,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test-02.csv",
            "output_path": "data/preprocessed/sequential-v2-02/test-",
            "load_from_pkl": False,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": True,
            "vector_size": 7500,
        }
    ),

    "sequential-conversation-v2-dataset-onehot-01": (
        "basic-sequential",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/train-01.csv",
            "output_path": "data/preprocessed/sequential-v2-01/",
            "load_from_pkl": False,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": True,
            "vector_size": 7500,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test-01.csv",
            "output_path": "data/preprocessed/sequential-v2-01/test-",
            "load_from_pkl": False,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": True,
            "vector_size": 7500,
        }
    ),
    # For super loss datasets
    ###############
    "conversation-with-index-v2-dataset-onehot-raw-balanced-04": (
        "conversation-bow-with-triple",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/conversation/balanced-train-v2-04.csv",
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
    # TOYS
    ############################################################################
    ############################################################################

    "toy-conversation-v2-dataset-onehot-raw": (
        "conversation-bow",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/conversation/toy-balanced-train-v2-04.csv",
            "output_path": "data/preprocessed/toy-conversation-v2/",
            "load_from_pkl": True,
            "preprocessings": ["rr", "idr"],
            "persist_data": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/conversation/toy-balanced-test-v2-04.csv",
            "output_path": "data/preprocessed/toy-conversation-v2/test-",
            "load_from_pkl": True,
            "preprocessings": ["rr", "idr"],
            "persist_data": True,
        }
    ),

    "toy-cnn-conversation-v2-dataset-onehot": (
        "cnn-conversation-bow",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/conversation/toy-balanced-train-v2-04.csv",
            "output_path": "data/preprocessed/toy-conversation-v2/",
            "load_from_pkl": False,
            "preprocessings": ["rr"],
            "persist_data": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/conversation/toy-balanced-test-v2-04.csv",
            "output_path": "data/preprocessed/toy-conversation-v2/test-",
            "load_from_pkl": False,
            "preprocessings": ["rr"],
            "persist_data": True,
        }
    ),

    "toy-sequential-conversation-v2-dataset-onehot": (
        "basic-sequential",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/toy-train-04.csv",
            "output_path": "data/preprocessed/toy-sequential-v2-04/",
            "load_from_pkl": True,
            "preprocessings": ["rr"],
            "persist_data": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/toy-test-04.csv",
            "output_path": "data/preprocessed/toy-sequential-v2-04/test-",
            "load_from_pkl": True,
            "preprocessings": ["rr"],
            "persist_data": True,
        }
    ),
}

sessions = {
    # "balanced-v2-04": {
    #     "model": "ann",
    #     "commands": [
    #         ("train", {
    #             "epoch_num": 75,
    #             "batch_size": 8,
    #             "weights_checkpoint_path": "",
    #             },
    #             {
    #                 "dataset": "conversation-v2-dataset-onehot-raw-balanced-04",
    #                 "rerun_splitting": False,
    #                 "persist_splits": True,
    #                 "load_splits_from": None,
    #                 "n_splits": 4,
    #             }
    #         ),
    #         # ("test", {"weights_checkpoint_path": r"output\05-23-2023-13-58-00-balanced-v2-04\ann-with-superloss\conversation-bow-with-triple\rr.idr-0.005-32.1-0.0\weights\best_model.pth"}, {"dataset": "conversation-with-index-v2-dataset-onehot-raw-balanced-04"}),
    #         ("test", dict(), {"dataset": "conversation-v2-dataset-onehot-raw-balanced-04"}),
    #         ("eval", {"path": '', "use_current_session": True}, dict()),
    #     ],
    #     "model_configs": {
    #         "dimension_list": list([32]),
    #         "dropout_list": [0.0],
    #         "activation": ("relu", dict()),
    #         # "loss_func": ("weighted-binary-cross-entropy", {"pos_weight": 1.8}),
    #         "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(0.75)}),
    #         "lr": 0.005,
    #         "module_session_path": "output",
    #         "session_path_include_time": True,
    #     },
    # },
### !!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # "rnn-balanced-v2-04": {
    #     "model": "base-rnn",
    #     "commands": [
    #         ("train", {
    #             "epoch_num": 100,
    #             "batch_size": 16,
    #             "weights_checkpoint_path": "",
    #             },
    #             {
    #                 "dataset": "sequential-conversation-v2-dataset-onehot",
    #                 "rerun_splitting": False,
    #                 "persist_splits": True,
    #                 "load_splits_from": None,
    #                 "n_splits": 5,
    #             }
    #         ),
    #         ("test", dict(), {"dataset": "sequential-conversation-v2-dataset-onehot"}),
    #         ("eval", {"path": '', "use_current_session": True}, dict()),
    #     ],
    #     "model_configs": {
    #         "activation": ("relu", dict()),
    #         # "loss_func": ("weighted-binary-cross-entropy", {"pos_weight": 2}),
    #         "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(1.2)}),
    #         # "loss_func": ("BCEW", {"reduction": "sum"}),
    #         "lr": 0.0005,
    #         'hidden_size': 512,
    #         'num_layers': 1,
    #         "module_session_path": "output",
    #         "session_path_include_time": True,
    #     },
    # },

    # "lstm-balanced-v2-04": {
    #     "model": "lstm",
    #     "commands": [
    #         ("train", {
    #             "epoch_num": 100,
    #             "batch_size": 16,
    #             "weights_checkpoint_path": "",
    #             },
    #             {
    #                 "dataset": "sequential-conversation-v2-dataset-onehot",
    #                 "rerun_splitting": False,
    #                 "persist_splits": True,
    #                 "load_splits_from": None,
    #                 "n_splits": 5,
    #             }
    #         ),
    #         ("test", {"weights_checkpoint_path": None}, {"dataset": "sequential-conversation-v2-dataset-onehot"}),
    #         ("eval", {"path": '', "use_current_session": True}, dict()),
    #     ],
    #     "model_configs": {
    #         "activation": ("relu", dict()),
    #         # "loss_func": ("weighted-binary-cross-entropy", {"pos_weight": 2}),
    #         "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(1.2)}),
    #         # "loss_func": ("BCEW", {"reduction": "sum"}),
    #         "lr": 0.0005,
    #         'hidden_size': 512,
    #         'num_layers': 1,
    #         "module_session_path": "output",
    #         "session_path_include_time": True,
    #     },
    # },

    # "balanced-v2-04": { # 7777777777777777777
    #     "model": "gru",
    #     "commands": [
    #         ("train", {
    #             "epoch_num": 100,
    #             "batch_size": 16,
    #             "weights_checkpoint_path": "",
    #             },
    #             {
    #                 "dataset": "sequential-conversation-v2-dataset-onehot",
    #                 "rerun_splitting": False,
    #                 "persist_splits": True,
    #                 "load_splits_from": None,
    #                 "n_splits": 5,
    #             }
    #         ),
    #         ("test", dict(), {"dataset": "sequential-conversation-v2-dataset-onehot"}),
    #         ("eval", {"path": '', "use_current_session": True}, dict()),
    #     ],
    #     "model_configs": {
    #         "activation": ("relu", dict()),
    #         # "loss_func": ("weighted-binary-cross-entropy", {"pos_weight": 2}),
    #         "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(1.2)}),
    #         # "loss_func": ("BCEW", {"reduction": "sum"}),
    #         "lr": 0.0005,
    #         'hidden_size': 512,
    #         'num_layers': 1,
    #         "module_session_path": "output",
    #         "session_path_include_time": True,
    #     },
    # },


    # "cnn-balanced-v2-04": {
    #     "model": "ebrahimi-cnn",
    #     "commands": [
    #         ("train", {
    #             "epoch_num": 100,
    #             "batch_size": 8,
    #             "weights_checkpoint_path": "",
    #             },
    #             {
    #                 "dataset": "cnn-conversation-v2-dataset-onehot",
    #                 "rerun_splitting": False,
    #                 "persist_splits": True,
    #                 "load_splits_from": None,
    #                 "n_splits": 5,
    #             }
    #         ),
    #         ("test", dict(), {"dataset": "cnn-conversation-v2-dataset-onehot"}),
    #         ("eval", {"path": '', "use_current_session": True}, dict()),
    #     ],
    #     "model_configs": {
    #         "activation": ("relu", dict()),
    #         "loss_func": ("weighted-binary-cross-entropy", {"pos_weight": 1.0}),
    #         # "loss_func": ("weighted-binary-cross-entropy", {"reduction": "sum"}),
    #         "lr": 0.01,
    #         "module_session_path": "output",
    #         "session_path_include_time": True,
    #     },
    # },
#############################################################
#############################################################
#############################################################
    # "lstm-balanced-v2-04-temporal": {
    #     "model": "lstm",
    #     "commands": [
    #         ("train", {
    #             "epoch_num": 40,
    #             "batch_size": 16,
    #             "weights_checkpoint_path": "",
    #             },
    #             {
    #                 "dataset": "temporal-sequential-conversation-v2-dataset-onehot-04",
    #                 "rerun_splitting": False,
    #                 "persist_splits": True,
    #                 "load_splits_from": None,
    #                 "n_splits": 5,
    #             }
    #         ),
    #         ("test", {"weights_checkpoint_path": None}, {"dataset": "temporal-sequential-conversation-v2-dataset-onehot-04"}),
    #         ("eval", {"path": '', "use_current_session": True}, dict()),
    #     ],
    #     "model_configs": {
    #         "activation": ("relu", dict()),
    #         # "loss_func": ("weighted-binary-cross-entropy", {"pos_weight": 2}),
    #         "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(2)}),
    #         # "loss_func": ("BCEW", {"reduction": "sum"}),
    #         "lr": 0.0005,
    #         'hidden_size': 1024,
    #         'num_layers': 1,
    #         "module_session_path": "output",
    #         "session_path_include_time": True,
    #     },
    # },

    # "lstm-balanced-v2-02-temporal": {
    #     "model": "lstm",
    #     "commands": [
    #         ("train", {
    #             "epoch_num": 40,
    #             "batch_size": 16,
    #             "weights_checkpoint_path": "",
    #             },
    #             {
    #                 "dataset": "temporal-sequential-conversation-v2-dataset-onehot-02",
    #                 "rerun_splitting": False,
    #                 "persist_splits": True,
    #                 "load_splits_from": None,
    #                 "n_splits": 5,
    #             }
    #         ),
    #         ("test", {"weights_checkpoint_path": None}, {"dataset": "temporal-sequential-conversation-v2-dataset-onehot-02"}),
    #         ("eval", {"path": '', "use_current_session": True}, dict()),
    #     ],
    #     "model_configs": {
    #         "activation": ("relu", dict()),
    #         # "loss_func": ("weighted-binary-cross-entropy", {"pos_weight": 2}),
    #         "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(2)}),
    #         # "loss_func": ("BCEW", {"reduction": "sum"}),
    #         "lr": 0.0005,
    #         'hidden_size': 1024,
    #         'num_layers': 1,
    #         "module_session_path": "output",
    #         "session_path_include_time": True,
    #     },
    # },

    "lstm-balanced-v2-02-temporal-realtest": {
        "model": "lstm",
        "commands": [
            # ("train", {
            #     "epoch_num": 40,
            #     "batch_size": 16,
            #     "weights_checkpoint_path": "",
            #     },
            #     {
            #         "dataset": "temporal-sequential-conversation-v2-dataset-onehot-03-realtest",
            #         "rerun_splitting": False,
            #         "persist_splits": True,
            #         "load_splits_from": None,
            #         "n_splits": 5,
            #     }
            # ),
            ("test", {"weights_checkpoint_path": r"output\06-01-2023-17-54-36-lstm-balanced-v2-04-temporal\lstm\temporal-sequential\psw.rr.idr-v13002-lr0.000500-h1024-l1\weights\best_model.pth"},
                {"dataset": "temporal-sequential-conversation-v2-dataset-onehot-03-realtest"}),
            ("eval", {"path": '', "use_current_session": True}, dict()),
        ],
        "model_configs": {
            "activation": ("relu", dict()),
            # "loss_func": ("weighted-binary-cross-entropy", {"pos_weight": 2}),
            "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(2.333)}),
            # "loss_func": ("BCEW", {"reduction": "sum"}),
            "lr": 0.0009,
            'hidden_size': 1024,
            'num_layers': 1,
            "module_session_path": "output",
            "session_path_include_time": True,
        },
    },
    ######################  

    # "lstm-balanced-v2-02": {
    #     "model": "lstm",
    #     "commands": [
    #         ("train", {
    #             "epoch_num": 15,
    #             "batch_size": 4,
    #             "weights_checkpoint_path": "",
    #             },
    #             {
    #                 "dataset": "sequential-conversation-v2-dataset-onehot-02",
    #                 "rerun_splitting": False,
    #                 "persist_splits": True,
    #                 "load_splits_from": None,
    #                 "n_splits": 5,
    #             }
    #         ),
    #         ("test", {"weights_checkpoint_path": None}, {"dataset": "sequential-conversation-v2-dataset-onehot-02"}),
    #         ("eval", {"path": '', "use_current_session": True}, dict()),
    #     ],
    #     "model_configs": {
    #         "activation": ("relu", dict()),
    #         # "loss_func": ("weighted-binary-cross-entropy", {"pos_weight": 2}),
    #         "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(4)}),
    #         # "loss_func": ("BCEW", {"reduction": "sum"}),
    #         "lr": 0.0005,
    #         'hidden_size': 512,
    #         'num_layers': 1,
    #         "module_session_path": "output",
    #         "session_path_include_time": True,
    #     },
    # },

    # "lstm-balanced-v2-01": {
    #     "model": "lstm",
    #     "commands": [
    #         ("train", {
    #             "epoch_num": 15,
    #             "batch_size": 4,
    #             "weights_checkpoint_path": "",
    #             },
    #             {
    #                 "dataset": "sequential-conversation-v2-dataset-onehot-01",
    #                 "rerun_splitting": False,
    #                 "persist_splits": True,
    #                 "load_splits_from": None,
    #                 "n_splits": 5,
    #             }
    #         ),
    #         ("test", {"weights_checkpoint_path": None}, {"dataset": "sequential-conversation-v2-dataset-onehot-01"}),
    #         ("eval", {"path": '', "use_current_session": True}, dict()),
    #     ],
    #     "model_configs": {
    #         "activation": ("relu", dict()),
    #         # "loss_func": ("weighted-binary-cross-entropy", {"pos_weight": 2}),
    #         "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(9)}),
    #         # "loss_func": ("BCEW", {"reduction": "sum"}),
    #         "lr": 0.0005,
    #         'hidden_size': 512,
    #         'num_layers': 1,
    #         "module_session_path": "output",
    #         "session_path_include_time": True,
    #     },
    # },
# #############################################################  
    # "toy-balanced-v2-04": {
    #     "model": "ann",
    #     "commands": [
    #         ("train", {
    #             "epoch_num": 75,
    #             "batch_size": 8,
    #             "weights_checkpoint_path": "",
    #             },
    #             {
    #                 "dataset": "toy-conversation-v2-dataset-onehot-raw",
    #                 "rerun_splitting": False,
    #                 "persist_splits": True,
    #                 "load_splits_from": None,
    #                 "n_splits": 5,
    #             }
    #         ),
    #         ("test", dict(), {"dataset": "toy-conversation-v2-dataset-onehot-raw"}),
    #         ("eval", {"path": '', "use_current_session": True}, dict()),
    #     ],
    #     "model_configs": {
    #         "dimension_list": list([32]),
    #         "dropout_list": [0.0],
    #         "activation": ("relu", dict()),
    #         # "loss_func": ("weighted-binary-cross-entropy", {"pos_weight": 1.8}),
    #         "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(1)}),
    #         "lr": 0.005,
    #         "module_session_path": "output",
    #         "session_path_include_time": True,
    #     },
    # },


    # "toy-rnn-balanced-v2-04": {
    #     "model": "base-rnn",
    #     "commands": [
    #         ("train", {
    #             "epoch_num": 100,
    #             "batch_size": 8,
    #             "weights_checkpoint_path": "",
    #             },
    #             {
    #                 "dataset": "toy-sequential-conversation-v2-dataset-onehot",
    #                 "rerun_splitting": False,
    #                 "persist_splits": True,
    #                 "load_splits_from": None,
    #                 "n_splits": 3,
    #             }
    #         ),
    #         ("test", dict(), {"dataset": "toy-sequential-conversation-v2-dataset-onehot"}),
    #         ("eval", {"path": '', "use_current_session": True}, dict()),
    #     ],
    #     "model_configs": {
    #         "activation": ("relu", dict()),
    #         # "loss_func": ("weighted-binary-cross-entropy", {"pos_weight": 2}),
    #         "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(0.66)}),
    #         # "loss_func": ("BCEW", {"reduction": "sum"}),
    #         "lr": 0.001,
    #         'hidden_size': 128,
    #         'num_layers': 1,
    #         "module_session_path": "output",
    #         "session_path_include_time": False,
    #     },
    # },


    # "toy-cnn-balanced-v2-04": {
    #     "model": "ebrahimi-cnn",
    #     "commands": [
    #         ("train", {
    #             "epoch_num": 100,
    #             "batch_size": 8,
    #             "weights_checkpoint_path": "",
    #             },
    #             {
    #                 "dataset": "toy-cnn-conversation-v2-dataset-onehot",
    #                 "rerun_splitting": False,
    #                 "persist_splits": False,
    #                 "load_splits_from": None,
    #                 "n_splits": 3,
    #             }
    #         ),
    #         ("test", dict(), {"dataset": "toy-cnn-conversation-v2-dataset-onehot"}),
    #         ("eval", {"path": '', "use_current_session": True}, dict()),
    #     ],
    #     "model_configs": {
    #         "activation": ("relu", dict()),
    #         "loss_func": ("weighted-binary-cross-entropy", {"pos_weight": 1.5}),
    #         # "loss_func": ("weighted-binary-cross-entropy", {"reduction": "sum"}),
    #         "lr": 0.1,
    #         "module_session_path": "output",
    #         "session_path_include_time": True,
    #     },
    # },

    
#################
    # "ann-conversation-onehot-dataset-v2-balanced-03-02-01": {
    #         "model": "ann",
    #         "commands": [
    #             ("train", {"epoch_num": 50, "batch_size": 16, "k_fold": 8, "weights_checkpoint_path": r"output/04-26-2023-15-26-28/ann-conversation-onehot-dataset-v2-balanced-03-02/ann/conversation-bow/rr-0.001-32.1-0.0/weights/best_model.pth"},
    #                 "conversation-v2-dataset-onehot-raw-balanced-01"),
    #             ("test", dict(), "conversation-v2-dataset-onehot-raw-balanced-01"),
    #             ("eval", {"path": '', "use_current_session": True}, None),
    #         ],
    #         "model_configs": {
    #             "dimension_list": list([32]),
    #             "dropout_list": [0.0],
    #             "activation": ("relu", dict()),
    #             "loss_func": ("weighted-binary-cross-entropy", {"pos_weight": 8}),
    #             # "loss_func": ("weighted-binary-cross-entropy", {"reduction": "sum"}),
    #             "lr": 0.0008,
    #             "module_session_path": "output",
    #             "session_path_include_time": True,
    #         },
    #     },
    
    # "ann-conversation-onehot-dataset-v2-balanced-02": (
    #     {
    #         "model": "ann",
    #         "commands": [
    #             ("train", {"epoch_num": 50, "batch_size": 16, "k_fold": 8}, "conversation-v2-dataset-onehot-raw-balanced-02"),
    #             ("test", dict(), "conversation-v2-dataset-onehot-raw-balanced-02"),
    #             ("eval", {"path": '', "use_current_session": True}, None),
    #         ],
    #         "model_configs": {
    #             "dimension_list": list([32]),
    #             "dropout_list": [0.0],
    #             "activation": ("relu", dict()),
    #             "loss_func": ("weighted-binary-cross-entropy", {"pos_weight": 3.5}),
    #             # "loss_func": ("weighted-binary-cross-entropy", {"reduction": "sum"}),
    #             "lr": 0.025,
    #             "module_session_path": "output",
    #             "session_path_include_time": True,
    #         },
    #     },
    # )
#############################################################
#############################################################
#############################################################
    # "ann-conversation-onehot-dataset-v2-cleaned-balanced-01": (
    #     {
    #         "model": "ann",
    #         "commands": [
    #             ("train", {"epoch_num": 50, "batch_size": 32, "k_fold": 5}, "conversation-v2-dataset-onehot-raw-balanced-01"),
    #             ("test", dict(), "conversation-v2-dataset-onehot-raw-balanced-01"),
    #             ("eval", {"path": '', "use_current_session": True}, None),
    #         ],
    #         "model_configs": {
    #             "dimension_list": list([64]),
    #             "dropout_list": [],
    #             "activation": ("relu", dict()),
    #             "loss_func": ("weighted-binary-cross-entropy", {"pos_weight": 5}),
    #             # "loss_func": ("weighted-binary-cross-entropy", {"reduction": "sum"}),
    #             "lr": 0.025,
    #             "module_session_path": "output",
    #             "session_path_include_time": False,
    #         },
    #     },
    #     {
    #         "model": "ann",
    #         "commands": [
    #             ("train", {"epoch_num": 50, "batch_size": 32, "k_fold": 5}, "conversation-v2-dataset-onehot-raw-balanced-01"),
    #             ("test", dict(), "conversation-v2-dataset-onehot-raw-balanced-01"),
    #             ("eval", {"path": '', "use_current_session": True}, None),
    #         ],
    #         "model_configs": {
    #             "dimension_list": list([32]),
    #             "dropout_list": [],
    #             "activation": ("relu", dict()),
    #             "loss_func": ("weighted-binary-cross-entropy", {"pos_weight": 5}),
    #             # "loss_func": ("weighted-binary-cross-entropy", {"reduction": "sum"}),
    #             "lr": 0.025,
    #             "module_session_path": "output",
    #             "session_path_include_time": False,
    #         },
    #     },
    #     {
    #         "model": "ann",
    #         "commands": [
    #             ("train", {"epoch_num": 50, "batch_size": 32, "k_fold": 5}, "conversation-v2-dataset-onehot-raw-balanced-01"),
    #             ("test", dict(), "conversation-v2-dataset-onehot-raw-balanced-01"),
    #             ("eval", {"path": '', "use_current_session": True}, None),
    #         ],
    #         "model_configs": {
    #             "dimension_list": list([128]),
    #             "dropout_list": [],
    #             "activation": ("relu", dict()),
    #             "loss_func": ("weighted-binary-cross-entropy", {"pos_weight": 5}),
    #             # "loss_func": ("weighted-binary-cross-entropy", {"reduction": "sum"}),
    #             "lr": 0.025,
    #             "module_session_path": "output",
    #             "session_path_include_time": False,
    #         },
    #     },
    # ),
    # "ann-conversation-onehot-dataset-v2-cleaned-balanced-02": (
    #     {
    #         "model": "ann",
    #         "commands": [
    #             ("train", {"epoch_num": 50, "batch_size": 32, "k_fold": 5}, "conversation-v2-dataset-onehot-raw-balanced-02"),
    #             ("test", dict(), "conversation-v2-dataset-onehot-raw-balanced-02"),
    #             ("eval", {"path": '', "use_current_session": True}, None),
    #         ],
    #         "model_configs": {
    #             "dimension_list": list([32]),
    #             "dropout_list": [],
    #             "activation": ("relu", dict()),
    #             "loss_func": ("weighted-binary-cross-entropy", {"pos_weight": 3}),
    #             # "loss_func": ("weighted-binary-cross-entropy", {"reduction": "sum"}),
    #             "lr": 0.025,
    #             "module_session_path": "output",
    #             "session_path_include_time": False,
    #         },
    #     },
    #     {
    #         "model": "ann",
    #         "commands": [
    #             ("train", {"epoch_num": 50, "batch_size": 32, "k_fold": 5}, "conversation-v2-dataset-onehot-raw-balanced-02"),
    #             ("test", dict(), "conversation-v2-dataset-onehot-raw-balanced-02"),
    #             ("eval", {"path": '', "use_current_session": True}, None),
    #         ],
    #         "model_configs": {
    #             "dimension_list": list([64]),
    #             "dropout_list": [],
    #             "activation": ("relu", dict()),
    #             "loss_func": ("weighted-binary-cross-entropy", {"pos_weight": 3}),
    #             # "loss_func": ("weighted-binary-cross-entropy", {"reduction": "sum"}),
    #             "lr": 0.025,
    #             "module_session_path": "output",
    #             "session_path_include_time": False,
    #         },
    #     },
    #     {
    #         "model": "ann",
    #         "commands": [
    #             ("train", {"epoch_num": 50, "batch_size": 32, "k_fold": 5}, "conversation-v2-dataset-onehot-raw-balanced-02"),
    #             ("test", dict(), "conversation-v2-dataset-onehot-raw-balanced-02"),
    #             ("eval", {"path": '', "use_current_session": True}, None),
    #         ],
    #         "model_configs": {
    #             "dimension_list": list([128]),
    #             "dropout_list": [],
    #             "activation": ("relu", dict()),
    #             "loss_func": ("weighted-binary-cross-entropy", {"pos_weight": 3}),
    #             # "loss_func": ("weighted-binary-cross-entropy", {"reduction": "sum"}),
    #             "lr": 0.025,
    #             "module_session_path": "output",
    #             "session_path_include_time": False,
    #         },
    #     },
    # ),
}


FILTERED_CONFIGS = {
    "session_path_include_time",
    "data_path",
    "output_path",
    "load_from_pkl",
    "preprocessings",
    "persist_data",
    "splitting_configs",
}

IGNORED_PARAM_RESET = {"activation", "loss_function"}

USE_CUDA_IF_AVAILABLE = True

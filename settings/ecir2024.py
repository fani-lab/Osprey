import torch

__preprocessings__ = ["pr", "sw", "rr", "idr"] ## Just to make it easier to change configurations

datasets = {
    ############## Sequential bag-of-words
    "sequential-conversation-dataset-bow": (
        "sequential-bow-convsize",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/train.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "vector_size": 13000,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "vector_size": 13000,
            "apply_record_filter": True,
        }
    ),

    "temporal-sequential-conversation-dataset-bow": (
        "time-sequential-bow-convsize",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/train.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "vector_size": 5000,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "vector_size": 5000,
            "apply_record_filter": True,
        }
    ),

    "temporal-nauthor-sequential-conversation-dataset-bow": (
        "time-nauthor-sequential",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/train.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "vector_size": 13000,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "vector_size": 13000,
            "apply_record_filter": True,
        }
    ),

    ############## Sequential Embedding distilroberta
    "sequential-conversation-v2-dataset-distilroberta": (
        "sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/train.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        }
    ),

    "temporal-sequential-conversation-v2-dataset-distilroberta": (
        "temporal-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/train.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        }
    ),

    "temporal-nauthor-sequential-conversation-v2-dataset-distilroberta": (
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/train.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        }
    ),

     ############## Sequential Embedding bert-base-uncased
     "toy-sequential-conversation-v2-dataset-bert": (
        "sequential-bert-base",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/toy-train.csv",
            "output_path": "data/preprocessed/sequential-v2/toy-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/toy-test.csv",
            "output_path": "data/preprocessed/sequential-v2/toy-test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        }
    ),

    "toy-temporal-sequential-conversation-v2-dataset-distilroberta": (
        "temporal-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/toy-train.csv",
            "output_path": "data/preprocessed/sequential-v2/toy-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/toy-test.csv",
            "output_path": "data/preprocessed/sequential-v2/toy-test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        }
    ),

    "sequential-conversation-v2-dataset-bert": (
        "sequential-bert-base",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/train.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        }
    ),

    "temporal-sequential-conversation-v2-dataset-bert": (
        "temporal-sequential-bert-base",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/train.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        }
    ),

    "temporal-nauthor-sequential-conversation-v2-dataset-bert": (
        "temporal-nauthor-sequential-bert-base",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/train.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        }
    ),

    ######################### All the messages as block of text, Embedding and bag-of-words
    "nauthor-conversation-dataset-bag-of-words": (
        "nauthor-bag-of-words-conversation",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/conversation/train.csv",
            "output_path": "data/preprocessed/conversation-dataset-v2/",
            "vector_size": 13000,
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/conversation/test.csv",
            "output_path": "data/preprocessed/conversation-dataset-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "vector_size": 13000,
            "apply_record_filter": True,
        }
    ), 

    "nauthor-conversation-dataset-distilroberta-v1": (
        "nauthor-conversation-distilroberta-v1",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/conversation/train.csv",
            "output_path": "data/preprocessed/conversation-dataset-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/conversation/test.csv",
            "output_path": "data/preprocessed/conversation-dataset-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        }
    ),

    "nauthor-conversation-dataset-bert": (
        "nauthor-conversation-bert",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/conversation/train.csv",
            "output_path": "data/preprocessed/conversation-dataset-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/conversation/test.csv",
            "output_path": "data/preprocessed/conversation-dataset-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        }
    ),

    "conversation-dataset-onehot": (
        "conversation-bow",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/conversation/train.csv",
            "output_path": "data/preprocessed/conversation-dataset-v2/",
            "vector_size": 13000,
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/conversation/test.csv",
            "output_path": "data/preprocessed/conversation-dataset-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "vector_size": 13000,
            "apply_record_filter": True,
        }
    ),

    "conversation-dataset-distilroberta": (
        "conversation-distilroberta-v1",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/conversation/train.csv",
            "output_path": "data/preprocessed/conversation-dataset-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/conversation/test.csv",
            "output_path": "data/preprocessed/conversation-dataset-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        }
    ),
    
    "conversation-dataset-bert": (
        "conversation-bert-base-uncased",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/conversation/train.csv",
            "output_path": "data/preprocessed/conversation-dataset-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/conversation/test.csv",
            "output_path": "data/preprocessed/conversation-dataset-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        }
    ),
}

sessions = {
    ############### Non recurrent models
    "svm-rbf-bow": {
        "model": "base-svm",
        "commands": [
            ("train", {
                # "epoch_num": 30,
                # "batch_size": 8,
                "weights_checkpoint_path": "",
                },
                {
                    "dataset": "conversation-dataset-onehot",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "data/preprocessed/conversation-dataset-v2/conversation-distilroberta-v1/ppr.sw.rr.idr-v768-filtered/splits-n3stratified.pkl",
                    "n_splits": 3,
                }
            ),
            ("test", {"weights_checkpoint_path": []}, {"dataset": "conversation-dataset-onehot"}),
            ("eval", {"path": '', "use_current_session": True}, {"dataset": "conversation-dataset-onehot"}),
        ],
        "model_configs": {
            "lr": 0,
            "module_session_path": "output-ecir2024",
            "session_path_include_time": False,
        },
    },

    "svm-rbf-bert": {
        "model": "base-svm",
        "commands": [
            ("train", {
                # "epoch_num": 30,
                # "batch_size": 8,
                "weights_checkpoint_path": "",
                },
                {
                    "dataset": "conversation-dataset-bert",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "data/preprocessed/conversation-dataset-v2/conversation-distilroberta-v1/ppr.sw.rr.idr-v768-filtered/splits-n3stratified.pkl",
                    "n_splits": 3,
                }
            ),
            ("test", {"weights_checkpoint_path": []}, {"dataset": "conversation-dataset-bert"}),
            ("eval", {"path": '', "use_current_session": True}, {"dataset": "conversation-dataset-bert"}),
        ],
        "model_configs": {
            "lr": 0,
            "module_session_path": "output-ecir2024",
            "session_path_include_time": False,
        },
    },

    "svm-rbf-distilroberta": {
        "model": "base-svm",
        "commands": [
            ("train", {
                # "epoch_num": 30,
                # "batch_size": 8,
                "weights_checkpoint_path": "",
                },
                {
                    "dataset": "conversation-dataset-distilroberta",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "data/preprocessed/conversation-dataset-v2/conversation-distilroberta-v1/ppr.sw.rr.idr-v768-filtered/splits-n3stratified.pkl",
                    "n_splits": 3,
                }
            ),
            ("test", {"weights_checkpoint_path": []}, {"dataset": "conversation-dataset-distilroberta"}),
            ("eval", {"path": '', "use_current_session": True}, {"dataset": "conversation-dataset-distilroberta"}),
        ],
        "model_configs": {
            "lr": 0,
            "module_session_path": "output-ecir2024",
            "session_path_include_time": False,
        },
    },

    "feedforward-bert": {
        "model": "ann",
        "commands": [
            ("train", {
                "epoch_num": 30,
                "batch_size": 8,
                "weights_checkpoint_path": "",
                },
                {
                    "dataset": "conversation-dataset-bert",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "data/preprocessed/conversation-dataset-v2/conversation-distilroberta-v1/ppr.sw.rr.idr-v768-filtered/splits-n3stratified.pkl",
                    "n_splits": 3,
                }
            ),
            ("test", {"weights_checkpoint_path": []}, {"dataset": "conversation-dataset-bert"}),
            ("eval", {"path": '', "use_current_session": True}, {"dataset": "conversation-dataset-bert"}),
        ],
        "model_configs": {
            "dimension_list": list([256]),
            "dropout_list": [0.0],
            "activation": ("relu", dict()),
            "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(16.5)}),
            "lr": 0.0005,
            "module_session_path": "output-ecir2024",
            "session_path_include_time": False,
            "early_stop": True,
        },
    },

    "feedforward-distilroberta": {
        "model": "ann",
        "commands": [
            ("train", {
                "epoch_num": 30,
                "batch_size": 8,
                "weights_checkpoint_path": "",
                },
                {
                    "dataset": "conversation-dataset-distilroberta",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "data/preprocessed/conversation-dataset-v2/conversation-distilroberta-v1/ppr.sw.rr.idr-v768-filtered/splits-n3stratified.pkl",
                    "n_splits": 3,
                }
            ),
            ("test", {"weights_checkpoint_path": []}, {"dataset": "conversation-dataset-distilroberta"}),
            ("eval", {"path": '', "use_current_session": True}, {"dataset": "conversation-dataset-distilroberta"}),
        ],
        "model_configs": {
            "dimension_list": list([256]),
            "dropout_list": [0.0],
            "activation": ("relu", dict()),
            "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(16.5)}),
            "lr": 0.0005,
            "module_session_path": "output-ecir2024",
            "session_path_include_time": False,
            "early_stop": True,
        },
    },

    "feedforward-bow": {
        "model": "ann",
        "commands": [
            ("train", {
                "epoch_num": 30,
                "batch_size": 8,
                "weights_checkpoint_path": "",
                },
                {
                    "dataset": "conversation-dataset-onehot",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "data/preprocessed/conversation-dataset-v2/conversation-distilroberta-v1/ppr.sw.rr.idr-v768-filtered/splits-n3stratified.pkl",
                    "n_splits": 3,
                }
            ),
            ("test", {"weights_checkpoint_path": []}, {"dataset": "conversation-dataset-onehot"}),
            ("eval", {"path": '', "use_current_session": True}, {"dataset": "conversation-dataset-onehot"}),
        ],
        "model_configs": {
            "dimension_list": list([256]),
            "dropout_list": [0.0],
            "activation": ("relu", dict()),
            "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(16.5)}),
            "lr": 0.0005,
            "module_session_path": "output-ecir2024",
            "session_path_include_time": False,
            "early_stop": True,
        },
    },

    "feedforward-32-bow": {
        "model": "ann",
        "commands": [
            ("train", {
                "epoch_num": 30,
                "batch_size": 8,
                "weights_checkpoint_path": "",
                },
                {
                    "dataset": "conversation-dataset-onehot",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "data/preprocessed/conversation-dataset-v2/conversation-distilroberta-v1/ppr.sw.rr.idr-v768-filtered/splits-n3stratified.pkl",
                    "n_splits": 3,
                }
            ),
            ("test", {"weights_checkpoint_path": []}, {"dataset": "conversation-dataset-onehot"}),
            ("eval", {"path": '', "use_current_session": True}, {"dataset": "conversation-dataset-onehot"}),
        ],
        "model_configs": {
            "dimension_list": list([32]),
            "dropout_list": [0.0],
            "activation": ("relu", dict()),
            "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(16.5)}),
            "lr": 0.0005,
            "module_session_path": "output-ecir2024",
            "session_path_include_time": False,
            "early_stop": True,
        },
    },

    "feedforward-block-of-text-nauthor": {
        "model": "ann",
        "commands": [
            ("train", {
                "epoch_num": 30,
                "batch_size": 8,
                "weights_checkpoint_path": "",
                }, # nauthor-conversation-dataset-bag-of-words, nauthor-conversation-dataset-distilroberta-v1, nauthor-conversation-dataset-bert
                {
                    "dataset": "nauthor-conversation-dataset-distilroberta-v1",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "data/preprocessed/conversation-dataset-v2/conversation-distilroberta-v1/ppr.sw.rr.idr-v768-filtered/splits-n3stratified.pkl",
                    "n_splits": 3,
                }
            ),
            ("test", {"weights_checkpoint_path": []}, {"dataset": "nauthor-conversation-dataset-distilroberta-v1"}),
            ("eval", {"path": '', "use_current_session": True}, {"dataset": "nauthor-conversation-dataset-distilroberta-v1"}),
        ],
        "model_configs": {
            "dimension_list": list([32]),
            "dropout_list": [0.0],
            "activation": ("relu", dict()),
            "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(16.5)}),
            "lr": 0.0005,
            "module_session_path": "output-ecir2024",
            "session_path_include_time": False,
            "early_stop": True,
        },
    },

    "svm-block-of-text-nauthor": {
        "model": "base-svm",
        "commands": [
            ("train", {
                "weights_checkpoint_path": "",
                }, # nauthor-conversation-dataset-bag-of-words, nauthor-conversation-dataset-distilroberta-v1, nauthor-conversation-dataset-bert
                {
                    "dataset": "nauthor-conversation-dataset-distilroberta-v1",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "data/preprocessed/conversation-dataset-v2/conversation-distilroberta-v1/ppr.sw.rr.idr-v768-filtered/splits-n3stratified.pkl",
                    "n_splits": 3,
                }
            ),
            ("test", {"weights_checkpoint_path": []}, {"dataset": "nauthor-conversation-dataset-distilroberta-v1"}),
            ("eval", {"path": '', "use_current_session": True}, {"dataset": "nauthor-conversation-dataset-distilroberta-v1"}),
        ],
        "model_configs": {
            "lr": 0,
            "module_session_path": "output-ecir2024",
            "session_path_include_time": False,
        },
    },

    ############### sequential distilroberta
    "lstm-distilroberta-temporal-nauthor": {
        "model": "lstm",
        "commands": [
            ("train", {
                "epoch_num": 30,
                "batch_size": 8,
                "weights_checkpoint_path": "",
                },
                {
                    "dataset": "temporal-nauthor-sequential-conversation-v2-dataset-distilroberta",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "data/splits-sequential-filtered-convsize-author2/splits-n3stratified.pkl",
                    "n_splits": 3,
                }
            ),
            ("test", {"weights_checkpoint_path": []}, {"dataset": "temporal-nauthor-sequential-conversation-v2-dataset-distilroberta"}),
            ("eval", {"path": '', "use_current_session": True}, {"dataset": "temporal-nauthor-sequential-conversation-v2-dataset-distilroberta"}),
        ],
        "model_configs": {
            "activation": ("relu", dict()),
            "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(16.5)}),
            "lr": 0.0005,
            'hidden_size': 514,
            'num_layers': 1,
            "module_session_path": "output-ecir2024",
            "session_path_include_time": False,
            "early_stop": True,
        },
    },

    "gru-distilroberta-temporal-nauthor": {
        "model": "gru",
        "commands": [
            ("train", {
                "epoch_num": 30,
                "batch_size": 8,
                "weights_checkpoint_path": "",
                },
                {
                    "dataset": "temporal-nauthor-sequential-conversation-v2-dataset-distilroberta",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "data/splits-sequential-filtered-convsize-author2/splits-n3stratified.pkl",
                    "n_splits": 3,
                }
            ),
            ("test", {"weights_checkpoint_path": []}, {"dataset": "temporal-nauthor-sequential-conversation-v2-dataset-distilroberta"}),
            ("eval", {"path": '', "use_current_session": True}, {"dataset": "temporal-nauthor-sequential-conversation-v2-dataset-distilroberta"}),
        ],
        "model_configs": {
            "activation": ("relu", dict()),
            "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(16.5)}),
            "lr": 0.0005,
            'hidden_size': 514,
            'num_layers': 1,
            "module_session_path": "output-ecir2024",
            "session_path_include_time": False,
            "early_stop": True,
        },
    },

    "rnn-distilroberta-temporal-nauthor": {
        "model": "base-rnn",
        "commands": [
            ("train", {
                "epoch_num": 30,
                "batch_size": 8,
                "weights_checkpoint_path": "",
                },
                {
                    "dataset": "temporal-nauthor-sequential-conversation-v2-dataset-distilroberta",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "data/splits-sequential-filtered-convsize-author2/splits-n3stratified.pkl",
                    "n_splits": 3,
                }
            ),
            ("test", {"weights_checkpoint_path": []}, {"dataset": "temporal-nauthor-sequential-conversation-v2-dataset-distilroberta"}),
            ("eval", {"path": '', "use_current_session": True}, {"dataset": "temporal-nauthor-sequential-conversation-v2-dataset-distilroberta"}),
        ],
        "model_configs": {
            "activation": ("relu", dict()),
            "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(16.5)}),
            "lr": 0.0005,
            'hidden_size': 514,
            'num_layers': 1,
            "module_session_path": "output-ecir2024",
            "session_path_include_time": False,
            "early_stop": True,
        },
    },

    ####################### sequential berts 
    "rnn-distilroberta-temporal-toy": {
        "model": "base-rnn",
        "commands": [
            ("train", {
                "epoch_num": 30,
                "batch_size": 8,
                "weights_checkpoint_path": "",
                },
                {
                    "dataset": "toy-temporal-sequential-conversation-v2-dataset-distilroberta",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "data/splits-sequential-filtered-convsize-author2/splits-n3stratified.pkl",
                    "n_splits": 3,
                }
            ),
            ("test", {"weights_checkpoint_path": []}, {"dataset": "toy-temporal-sequential-conversation-v2-dataset-distilroberta"}),
            ("eval", {"path": '', "use_current_session": True}, {"dataset": "toy-temporal-sequential-conversation-v2-dataset-distilroberta"}),
        ],
        "model_configs": {
            "activation": ("relu", dict()),
            "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(16.5)}),
            "lr": 0.0005,
            'hidden_size': 1024,
            'num_layers': 1,
            "module_session_path": "output-ecir2024",
            "session_path_include_time": False,
            "early_stop": True,
        },
    },
    "rnn-bert-toy": {
        "model": "base-rnn",
        "commands": [
            ("train", {
                "epoch_num": 30,
                "batch_size": 8,
                "weights_checkpoint_path": "",
                },
                {
                    "dataset": "toy-sequential-conversation-v2-dataset-bert",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "data/splits-sequential-filtered-convsize-author2/splits-n3stratified.pkl",
                    "n_splits": 3,
                }
            ),
            ("test", {"weights_checkpoint_path": []}, {"dataset": "toy-sequential-conversation-v2-dataset-bert"}),
            ("eval", {"path": '', "use_current_session": True}, {"dataset": "toy-sequential-conversation-v2-dataset-bert"}),
        ],
        "model_configs": {
            "activation": ("relu", dict()),
            "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(16.5)}),
            "lr": 0.0005,
            'hidden_size': 1024,
            'num_layers': 1,
            "module_session_path": "output-ecir2024",
            "session_path_include_time": False,
            "early_stop": True,
        },
    },
    "rnn-bert-temporal-nauthor": {
        "model": "base-rnn",
        "commands": [
            ("train", {
                "epoch_num": 30,
                "batch_size": 8,
                "weights_checkpoint_path": "",
                },
                {
                    "dataset": "temporal-nauthor-sequential-conversation-v2-dataset-bert",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "data/splits-sequential-filtered-convsize-author2/splits-n3stratified.pkl",
                    "n_splits": 3,
                }
            ),
            ("test", {"weights_checkpoint_path": []}, {"dataset": "temporal-nauthor-sequential-conversation-v2-dataset-bert"}),
            ("eval", {"path": '', "use_current_session": True}, {"dataset": "temporal-nauthor-sequential-conversation-v2-dataset-bert"}),
        ],
        "model_configs": {
            "activation": ("relu", dict()),
            "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(16.5)}),
            "lr": 0.0005,
            'hidden_size': 512,
            'num_layers': 1,
            "module_session_path": "output-ecir2024",
            "session_path_include_time": False,
            "early_stop": True,
        },
    },

    "gru-bert-temporal-nauthor": {
        "model": "gru",
        "commands": [
            ("train", {
                "epoch_num": 30,
                "batch_size": 8,
                "weights_checkpoint_path": "",
                },
                {
                    "dataset": "temporal-nauthor-sequential-conversation-v2-dataset-bert",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "data/splits-sequential-filtered-convsize-author2/splits-n3stratified.pkl",
                    "n_splits": 3,
                }
            ),
            ("test", {"weights_checkpoint_path": []}, {"dataset": "temporal-nauthor-sequential-conversation-v2-dataset-bert"}),
            ("eval", {"path": '', "use_current_session": True}, {"dataset": "temporal-nauthor-sequential-conversation-v2-dataset-bert"}),
        ],
        "model_configs": {
            "activation": ("relu", dict()),
            "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(16.5)}),
            "lr": 0.0005,
            'hidden_size': 512,
            'num_layers': 1,
            "module_session_path": "output-ecir2024",
            "session_path_include_time": False,
            "early_stop": True,
        },
    },

    "lstm-bert-temporal-nauthor": {
        "model": "lstm",
        "commands": [
            ("train", {
                "epoch_num": 30,
                "batch_size": 8,
                "weights_checkpoint_path": "",
                },
                {
                    "dataset": "temporal-nauthor-sequential-conversation-v2-dataset-bert",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "data/splits-sequential-filtered-convsize-author2/splits-n3stratified.pkl",
                    "n_splits": 3,
                }
            ),
            ("test", {"weights_checkpoint_path": []}, {"dataset": "temporal-nauthor-sequential-conversation-v2-dataset-bert"}),
            ("eval", {"path": '', "use_current_session": True}, {"dataset": "temporal-nauthor-sequential-conversation-v2-dataset-bert"}),
        ],
        "model_configs": {
            "activation": ("relu", dict()),
            "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(16.5)}),
            "lr": 0.0005,
            'hidden_size': 512,
            'num_layers': 1,
            "module_session_path": "output-ecir2024",
            "session_path_include_time": False,
            "early_stop": True,
        },
    },

    ##################### sequential bag-of-words
    "lstm-bow-temporal-nauthor": {
        "model": "lstm",
        "commands": [
            ("train", {
                "epoch_num": 30,
                "batch_size": 8,
                "weights_checkpoint_path": "",
                },
                {
                    "dataset": "temporal-nauthor-sequential-conversation-dataset-bow",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "data/splits-sequential-filtered-convsize-author2/splits-n3stratified.pkl",
                    "n_splits": 3,
                }
            ),
            ("test", {"weights_checkpoint_path": []}, {"dataset": "temporal-nauthor-sequential-conversation-dataset-bow"}),
            ("eval", {"path": '', "use_current_session": True}, {"dataset": "temporal-nauthor-sequential-conversation-dataset-bow"}),
        ],
        "model_configs": {
            "activation": ("relu", dict()),
            "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(16.5)}),
            "lr": 0.0005,
            'hidden_size': 512,
            'num_layers': 1,
            "module_session_path": "output-ecir2024",
            "session_path_include_time": False,
            "early_stop": True,
        },
    },
    
    "lstm-bow-temporal": {
        "model": "lstm",
        "commands": [
            ("train", {
                "epoch_num": 30,
                "batch_size": 8,
                "weights_checkpoint_path": "",
                },
                {
                    "dataset": "temporal-sequential-conversation-dataset-bow",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "data/splits-sequential-filtered-convsize-author2/splits-n3stratified.pkl",
                    "n_splits": 3,
                }
            ),
            ("test", {"weights_checkpoint_path": []}, {"dataset": "temporal-sequential-conversation-dataset-bow"}),
            ("eval", {"path": '', "use_current_session": True}, {"dataset": "temporal-sequential-conversation-dataset-bow"}),
        ],
        "model_configs": {
            "activation": ("relu", dict()),
            "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(16.5)}),
            "lr": 0.0005,
            'hidden_size': 1024,
            'num_layers': 1,
            "module_session_path": "output-ecir2024",
            "session_path_include_time": False,
            "early_stop": True,
        },
    },

    "gru-bow-temporal-nauthor": {
        "model": "gru",
        "commands": [
            ("train", {
                "epoch_num": 30,
                "batch_size": 8,
                "weights_checkpoint_path": "",
                },
                {
                    "dataset": "temporal-nauthor-sequential-conversation-dataset-bow",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "data/splits-sequential-filtered-convsize-author2/splits-n3stratified.pkl",
                    "n_splits": 3,
                }
            ),
            ("test", {"weights_checkpoint_path": []}, {"dataset": "temporal-nauthor-sequential-conversation-dataset-bow"}),
            ("eval", {"path": '', "use_current_session": True}, {"dataset": "temporal-nauthor-sequential-conversation-dataset-bow"}),
        ],
        "model_configs": {
            "activation": ("relu", dict()),
            "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(16.5)}),
            "lr": 0.0005,
            'hidden_size': 512,
            'num_layers': 1,
            "module_session_path": "output-ecir2024",
            "session_path_include_time": False,
            "early_stop": True,
        },
    },

    "rnn-bow-temporal-nauthor": {
        "model": "base-rnn",
        "commands": [
            ("train", {
                "epoch_num": 30,
                "batch_size": 8,
                "weights_checkpoint_path": "",
                },
                {
                    "dataset": "temporal-nauthor-sequential-conversation-dataset-bow",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "data/splits-sequential-filtered-convsize-author2/splits-n3stratified.pkl",
                    "n_splits": 3,
                }
            ),
            ("test", {"weights_checkpoint_path": []}, {"dataset": "temporal-nauthor-sequential-conversation-dataset-bow"}),
            ("eval", {"path": '', "use_current_session": True}, {"dataset": "temporal-nauthor-sequential-conversation-dataset-bow"}),
        ],
        "model_configs": {
            "activation": ("relu", dict()),
            "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(16.5)}),
            "lr": 0.0005,
            'hidden_size': 512,
            'num_layers': 1,
            "module_session_path": "output-ecir2024",
            "session_path_include_time": False,
            "early_stop": True,
        },
    },
}

USE_CUDA_IF_AVAILABLE = True

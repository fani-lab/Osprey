import torch

__preprocessings__ = [] ## Just to make it easier to change configurations

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

    "temporal-nauthor-sequential-conversation-dataset-bow": (
        "time-nauthor-sequential-bow-convsize",  # short name of the dataset
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

    ######################### All the messages as block of text
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
}

sessions = {
    ############### Non recurrent models
    "feedforward-256-bow": {
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
                    "load_splits_from": "data/splits-sequential-filtered-convsize-author2/splits-n3stratified.pkl",
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
                    "rerun_splitting": True,
                    "persist_splits": True,
                    "load_splits_from": "",
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

    ############### sequential distilroberta
    "lstm-distilroberta-temporal": {
        "model": "lstm",
        "commands": [
            ("train", {
                "epoch_num": 30,
                "batch_size": 8,
                "weights_checkpoint_path": "",
                },
                {
                    "dataset": "temporal-sequential-conversation-v2-dataset-distilroberta",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "data/splits-sequential-filtered-convsize-author2/splits-n3stratified.pkl",
                    "n_splits": 3,
                }
            ),
            ("test", {"weights_checkpoint_path": []}, {"dataset": "temporal-sequential-conversation-v2-dataset-distilroberta"}),
            ("eval", {"path": '', "use_current_session": True}, {"dataset": "temporal-sequential-conversation-v2-dataset-distilroberta"}),
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

    "gru-distilroberta-temporal": {
        "model": "gru",
        "commands": [
            ("train", {
                "epoch_num": 30,
                "batch_size": 8,
                "weights_checkpoint_path": "",
                },
                {
                    "dataset": "temporal-sequential-conversation-v2-dataset-distilroberta",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "data/splits-sequential-filtered-convsize-author2/splits-n3stratified.pkl",
                    "n_splits": 3,
                }
            ),
            ("test", {"weights_checkpoint_path": []}, {"dataset": "temporal-sequential-conversation-v2-dataset-distilroberta"}),
            ("eval", {"path": '', "use_current_session": True}, {"dataset": "temporal-sequential-conversation-v2-dataset-distilroberta"}),
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

    "rnn-distilroberta-temporal": {
        "model": "base-rnn",
        "commands": [
            ("train", {
                "epoch_num": 30,
                "batch_size": 8,
                "weights_checkpoint_path": "",
                },
                {
                    "dataset": "temporal-sequential-conversation-v2-dataset-distilroberta",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "data/splits-sequential-filtered-convsize-author2/splits-n3stratified.pkl",
                    "n_splits": 3,
                }
            ),
            ("test", {"weights_checkpoint_path": []}, {"dataset": "temporal-sequential-conversation-v2-dataset-distilroberta"}),
            ("eval", {"path": '', "use_current_session": True}, {"dataset": "temporal-sequential-conversation-v2-dataset-distilroberta"}),
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
    "rnn-bert-temporal": {
        "model": "base-rnn",
        "commands": [
            ("train", {
                "epoch_num": 30,
                "batch_size": 8,
                "weights_checkpoint_path": "",
                },
                {
                    "dataset": "temporal-sequential-conversation-v2-dataset-bert",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "data/splits-sequential-filtered-convsize-author2/splits-n3stratified.pkl",
                    "n_splits": 3,
                }
            ),
            ("test", {"weights_checkpoint_path": []}, {"dataset": "temporal-sequential-conversation-v2-dataset-bert"}),
            ("eval", {"path": '', "use_current_session": True}, {"dataset": "temporal-sequential-conversation-v2-dataset-bert"}),
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

    "gru-bert-temporal": {
        "model": "gru",
        "commands": [
            ("train", {
                "epoch_num": 30,
                "batch_size": 8,
                "weights_checkpoint_path": "",
                },
                {
                    "dataset": "temporal-sequential-conversation-v2-dataset-bert",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "data/splits-sequential-filtered-convsize-author2/splits-n3stratified.pkl",
                    "n_splits": 3,
                }
            ),
            ("test", {"weights_checkpoint_path": []}, {"dataset": "temporal-sequential-conversation-v2-dataset-bert"}),
            ("eval", {"path": '', "use_current_session": True}, {"dataset": "temporal-sequential-conversation-v2-dataset-bert"}),
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

    "lstm-bert-temporal": {
        "model": "lstm",
        "commands": [
            ("train", {
                "epoch_num": 30,
                "batch_size": 8,
                "weights_checkpoint_path": "",
                },
                {
                    "dataset": "temporal-sequential-conversation-v2-dataset-bert",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "data/splits-sequential-filtered-convsize-author2/splits-n3stratified.pkl",
                    "n_splits": 3,
                }
            ),
            ("test", {"weights_checkpoint_path": []}, {"dataset": "temporal-sequential-conversation-v2-dataset-bert"}),
            ("eval", {"path": '', "use_current_session": True}, {"dataset": "temporal-sequential-conversation-v2-dataset-bert"}),
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

    ##################### sequential bag-of-words
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

    "gru-bow-temporal": {
        "model": "gru",
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

    "rnn-bow-temporal": {
        "model": "base-rnn",
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
}

USE_CUDA_IF_AVAILABLE = True

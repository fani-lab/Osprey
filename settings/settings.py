import torch


datasets = {

    "bert-based-uncased-tokenized-toy": (
        "bert-based-uncased",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/toy-train.csv",
            "output_path": "data/preprocessed/transformer/toy-",
            "load_from_pkl": False,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": False,
            "apply_record_filter": False,
        },
        {      # test configs
            "data_path": "data/dataset-v2/toy-test.csv",
            "output_path": "data/preprocessed/transformer/toy-test-",
            "load_from_pkl": False,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": False,
            "apply_record_filter": False,
        }
    ),

    "bert-based-uncased-tokenized": (
        "bert-based-uncased",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/train.csv",
            "output_path": "data/preprocessed/transformer/",
            "load_from_pkl": True,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": True,
            "apply_record_filter": False,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "output_path": "data/preprocessed/transformer/test-",
            "load_from_pkl": True,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": True,
            "apply_record_filter": False,
        }
    ),

################################################

    "finetuning-v2-dataset": (
        "finetuning-bert",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/train.csv",
            "output_path": "data/embeddings/finetuned/",
            "load_from_pkl": True,
            "preprocessings": ["sw", "rr", "bert_idr"],
            "persist_data": True,
            "apply_record_filter": False,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "output_path": "data/embeddings/finetuned/test-",
            "load_from_pkl": True,
            "preprocessings": ["sw", "rr", "bert_idr"],
            "persist_data": True,
            "apply_record_filter": False,
        }
    ),

    "finetuning-v2-dataset-toy": (
        "finetuning-bert",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/toy-train.csv",
            "output_path": "data/embeddings/finetuned/toy-",
            "load_from_pkl": True,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": True,
            "apply_record_filter": False,
        },
        {      # test configs
            "data_path": "data/dataset-v2/toy-test.csv",
            "output_path": "data/embeddings/finetuned/toy-test-",
            "load_from_pkl": True,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": True,
            "apply_record_filter": False,
        }
    ),

    ##################

    "temporal-sequential-conversation-v2-dataset-onehot-allreal": (
        "temporal-sequential",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/train.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": True,
            "vector_size": 13000,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": True,
            "vector_size": 13000,
            "apply_record_filter": False,
        }
    ),

    "temporal-nauthor-sequential-conversation-v2-dataset-onehot-allreal": (
        "time-nauthor-sequential",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/train.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": True,
            "vector_size": 13000,
            "apply_record_filter": False,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": True,
            "vector_size": 13000,
            "apply_record_filter": False,
        }
    ),

    "temporal-nauthor-sequential-conversation-v2-dataset-onehot-toy": (
        "time-nauthor-sequential",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/toy-train.csv",
            "output_path": "data/preprocessed/sequential-v2/toy-",
            "load_from_pkl": False,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": True,
            "vector_size": 13000,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/toy-test.csv",
            "output_path": "data/preprocessed/sequential-v2/toy-test-",
            "load_from_pkl": False,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": True,
            "vector_size": 13000,
            "apply_record_filter": False,
        }
    ),


    ########## temporal sequentials
    "temporal-sequential-conversation-v2-dataset-onehot-04": (
        "temporal-sequential",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/train-04.csv",
            "output_path": "data/preprocessed/sequential-v2-04/",
            "load_from_pkl": True,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": True,
            "vector_size": 13000,
            "apply_record_filter": False,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test-04.csv",
            "output_path": "data/preprocessed/sequential-v2-04/test-",
            "load_from_pkl": True,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": True,
            "vector_size": 13000,
            "apply_record_filter": False,
        }
    ),

    # "temporal-sequential-conversation-v2-dataset-onehot-03": (
    #     "temporal-sequential",  # short name of the dataset
    #     {       # train configs
    #         "data_path": "data/dataset-v2/train-03.csv",
    #         "output_path": "data/preprocessed/sequential-v2-03/",
    #         "load_from_pkl": True,
    #         "preprocessings": ["sw", "rr", "idr"],
    #         "persist_data": True,
    #         "vector_size": 13000,
    #         "apply_record_filter": True,
    #     },
    #     {      # test configs
    #         "data_path": "data/dataset-v2/test-03.csv",
    #         "output_path": "data/preprocessed/sequential-v2-03/test-",
    #         "load_from_pkl": True,
    #         "preprocessings": ["sw", "rr", "idr"],
    #         "persist_data": True,
    #         "vector_size": 13000,
    #         "apply_record_filter": False,
    #     }
    # ),

    # "temporal-sequential-conversation-v2-dataset-onehot-02": (
    #     "temporal-sequential",  # short name of the dataset
    #     {       # train configs
    #         "data_path": "data/dataset-v2/train-02.csv",
    #         "output_path": "data/preprocessed/sequential-v2-02/",
    #         "load_from_pkl": True,
    #         "preprocessings": ["sw", "rr", "idr"],
    #         "persist_data": True,
    #         "vector_size": 11000,
    #         "apply_record_filter": True,
    #     },
    #     {      # test configs
    #         "data_path": "data/dataset-v2/test-02.csv",
    #         "output_path": "data/preprocessed/sequential-v2-02/test-",
    #         "load_from_pkl": True,
    #         "preprocessings": ["sw", "rr", "idr"],
    #         "persist_data": True,
    #         "vector_size": 11000,
    #         "apply_record_filter": False,
    #     }
    # ),

    # "temporal-sequential-conversation-v2-dataset-onehot-01": (
    #     "temporal-sequential",  # short name of the dataset
    #     {       # train configs
    #         "data_path": "data/dataset-v2/train-01.csv",
    #         "output_path": "data/preprocessed/sequential-v2-01/",
    #         "load_from_pkl": True,
    #         "preprocessings": ["sw", "rr", "idr"],
    #         "persist_data": True,
    #         "vector_size": 13000,
    #          "apply_record_filter": True,
    #     },
    #     {      # test configs
    #         "data_path": "data/dataset-v2/test-01.csv",
    #         "output_path": "data/preprocessed/sequential-v2-01/test-",
    #         "load_from_pkl": True,
    #         "preprocessings": ["sw", "rr", "idr"],
    #         "persist_data": True,
    #         "vector_size": 13000,
    #         "apply_record_filter": False,
    #     }
    # ),

    ########## temporal sequentials with realtests
    "temporal-sequential-conversation-v2-dataset-onehot-04-realtest": (
        "temporal-sequential",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/train-04.csv",
            "output_path": "data/preprocessed/sequential-v2-04/",
            "load_from_pkl": True,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": False,
            "vector_size": 13000,
            "apply_record_filter": False,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": True,
            "vector_size": 13000,
            "apply_record_filter": False,
        }
    ),
    "temporal-sequential-conversation-v2-dataset-onehot-03-realtest": (
        "temporal-sequential",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/train-03.csv",
            "output_path": "data/preprocessed/sequential-v2-03/",
            "load_from_pkl": True,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": True,
            "vector_size": 13000,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": True,
            "vector_size": 13000,
            "apply_record_filter": False,
        }
    ),

    "temporal-sequential-conversation-v2-dataset-onehot-02-realtest": (
        "temporal-sequential",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/train-02.csv",
            "output_path": "data/preprocessed/sequential-v2-02/",
            "load_from_pkl": True,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": True,
            "vector_size": 13000,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": True,
            "vector_size": 13000,
            "apply_record_filter": False,
        }
    ),

    "temporal-sequential-conversation-v2-dataset-onehot-01-realtest": (
        "temporal-sequential",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/train-01.csv",
            "output_path": "data/preprocessed/sequential-v2-01/",
            "load_from_pkl": True,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": True,
            "vector_size": 13000,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": ["sw", "rr", "idr"],
            "persist_data": True,
            "vector_size": 13000,
            "apply_record_filter": False,
        }
    ),
############################### 
    "toy-conversation-v2-dataset-onehot": (
        "conversation-bow",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/conversation/toy-train.csv",
            "output_path": "data/preprocessed/conversation-dataset-v2/toy-",
            "load_from_pkl": True,
            "preprocessings": ["pr", "sw", "rr"],
            "persist_data": True,
            "vector_size": 13000,
            "apply_record_filter": False,
        },
        {      # test configs
            "data_path": "data/dataset-v2/conversation/toy-test.csv",
            "output_path": "data/preprocessed/conversation-dataset-v2/toy-test-",
            "load_from_pkl": True,
            "preprocessings": ["pr", "sw", "rr"],
            "persist_data": True,
            "vector_size": 13000,
            "apply_record_filter": False,
        }
    ),
}




sessions = {
    # "lstm-balanced-v2-temporal": {
    #     "model": "lstm",
    #     "commands": [
    #         ("train", {
    #             "epoch_num": 20,
    #             "batch_size": 8,
    #             "weights_checkpoint_paths": "",
    #             },
    #             {
    #                 "dataset": "temporal-sequential-conversation-v2-dataset-onehot-allreal",
    #                 "rerun_splitting": False,
    #                 "persist_splits": True,
    #                 "load_splits_from": None,
    #                 "n_splits": 5,
    #             }
    #         ),
    #         ("test", {"weights_checkpoint_paths": ""}, {"dataset": "temporal-sequential-conversation-v2-dataset-onehot-allreal"}),
    #         # ("test", {"weights_checkpoint_paths": r"output/06-06-2023-16-17-29-lstm-balanced-v2-04-temporal/lstm/temporal-sequential/psw.rr.idr-v13000-nofilter-lr0.000500-h2056-l1/weights/f1/model_fold1.pth"}, {"dataset": "temporal-sequential-conversation-v2-dataset-onehot-04-realtest"}),
    #         ("eval", {"path": '', "use_current_session": True}, dict()),
    #     ],
    #     "model_configs": {
    #         "activation": ("relu", dict()),
    #         # "loss_func": ("weighted-binary-cross-entropy", {"pos_weight": 2}),
    #         "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(30)}),
    #         # "loss_func": ("BCEW", {"reduction": "sum"}),
    #         "lr": 0.0005,
    #         'hidden_size': 512,
    #         'num_layers': 1,
    #         "module_session_path": "output",
    #         "session_path_include_time": False,
    #     },
    # },
    # "lstm-balanced-v2-time-nauthor": {
    #     "model": "lstm",
    #     "commands": [
    #         ("train", {
    #             "epoch_num": 25,
    #             "batch_size": 16,
    #             "weights_checkpoint_paths": "",
    #             },
    #             {
    #                 "dataset": "temporal-nauthor-sequential-conversation-v2-dataset-onehot-allreal",
    #                 "rerun_splitting": False,
    #                 "persist_splits": True,
    #                 "load_splits_from": None,
    #                 "n_splits": 5,
    #             }
    #         ),
    #         ("test", {"weights_checkpoint_paths": ""}, {"dataset": "temporal-nauthor-sequential-conversation-v2-dataset-onehot-allreal"}),
    #         # ("test", {"weights_checkpoint_paths": r"output/06-06-2023-16-17-29-lstm-balanced-v2-04-temporal/lstm/temporal-sequential/psw.rr.idr-v13000-nofilter-lr0.000500-h2056-l1/weights/f1/model_fold1.pth"}, {"dataset": "temporal-sequential-conversation-v2-dataset-onehot-04-realtest"}),
    #         ("eval", {"path": '', "use_current_session": True}, dict()),
    #     ],
    #     "model_configs": {
    #         "activation": ("relu", dict()),
    #         # "loss_func": ("weighted-binary-cross-entropy", {"pos_weight": 2}),
    #         "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(30)}),
    #         # "loss_func": ("BCEW", {"reduction": "sum"}),
    #         "lr": 0.0005,
    #         'hidden_size': 2048,
    #         'num_layers': 1,
    #         "module_session_path": "output",
    #         "session_path_include_time": False,
    #     },
    # },

    # "toy-lstm-balanced-v2-time-nauthor": {
    #     "model": "lstm",
    #     "commands": [
    #         ("train", {
    #             "epoch_num": 20,
    #             "batch_size": 8,
    #             "weights_checkpoint_paths": "",
    #             },
    #             {
    #                 "dataset": "temporal-nauthor-sequential-conversation-v2-dataset-onehot-toy",
    #                 "rerun_splitting": False,
    #                 "persist_splits": True,
    #                 "load_splits_from": None,
    #                 "n_splits": 5,
    #             }
    #         ),
    #         ("test", {"weights_checkpoint_paths": ""}, {"dataset": "temporal-nauthor-sequential-conversation-v2-dataset-onehot-toy"}),
    #         # ("test", {"weights_checkpoint_paths": r"output/06-06-2023-16-17-29-lstm-balanced-v2-04-temporal/lstm/temporal-sequential/psw.rr.idr-v13000-nofilter-lr0.000500-h2056-l1/weights/f1/model_fold1.pth"}, {"dataset": "temporal-sequential-conversation-v2-dataset-onehot-04-realtest"}),
    #         ("eval", {"path": '', "use_current_session": True}, dict()),
    #     ],
    #     "model_configs": {
    #         "activation": ("relu", dict()),
    #         # "loss_func": ("weighted-binary-cross-entropy", {"pos_weight": 2}),
    #         "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(30)}),
    #         # "loss_func": ("BCEW", {"reduction": "sum"}),
    #         "lr": 0.0005,
    #         'hidden_size': 512,
    #         'num_layers': 1,
    #         "module_session_path": "output",
    #         "session_path_include_time": False,
    #     },
    # },
######################### original test sets with balanced train sets
    # "lstm-balanced-v2-04-temporal-realtest": {
    #     "model": "lstm",
    #     "commands": [
    #         # ("train", {
    #         #     "epoch_num": 20,
    #         #     "batch_size": 16,
    #         #     "weights_checkpoint_paths": "",
    #         #     },
    #         #     {
    #         #         "dataset": "temporal-sequential-conversation-v2-dataset-onehot-04-realtest",
    #         #         "rerun_splitting": False,
    #         #         "persist_splits": True,
    #         #         "load_splits_from": None,
    #         #         "n_splits": 5,
    #         #     }
    #         # ),
    #         # ("test", {"weights_checkpoint_paths": ""}, {"dataset": "temporal-sequential-conversation-v2-dataset-onehot-04-realtest"}),
    #         ("test", {"weights_checkpoint_paths": r"output/06-06-2023-16-17-29-lstm-balanced-v2-04-temporal/lstm/temporal-sequential/psw.rr.idr-v13000-nofilter-lr0.000500-h2056-l1/weights/f1/model_fold1.pth"}, {"dataset": "temporal-sequential-conversation-v2-dataset-onehot-04-realtest"}),
    #         ("eval", {"path": '', "use_current_session": True}, dict()),
    #     ],
    #     "model_configs": {
    #         "activation": ("relu", dict()),
    #         # "loss_func": ("weighted-binary-cross-entropy", {"pos_weight": 2}),
    #         "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(2)}),
    #         # "loss_func": ("BCEW", {"reduction": "sum"}),
    #         "lr": 0.0005,
    #         'hidden_size': 2056,
    #         'num_layers': 1,
    #         "module_session_path": "output",
    #         "session_path_include_time": True,
    #     },
    # },

    # "lstm-balanced-v2-02-temporal-realtest": {
    #     "model": "lstm",
    #     "commands": [
    #         ("train", {
    #             "epoch_num": 20,
    #             "batch_size": 16,
    #             "weights_checkpoint_paths": "",
    #             },
    #             {
    #                 "dataset": "temporal-sequential-conversation-v2-dataset-onehot-02-realtest",
    #                 "rerun_splitting": False,
    #                 "persist_splits": True,
    #                 "load_splits_from": None,
    #                 "n_splits": 5,
    #             }
    #         ),
    #         ("test", {"weights_checkpoint_paths": ""}, {"dataset": "temporal-sequential-conversation-v2-dataset-onehot-02-realtest"}),
    #         # ("test", {"weights_checkpoint_paths": r"output/06-06-2023-16-17-29-lstm-balanced-v2-04-temporal/lstm/temporal-sequential/psw.rr.idr-v13000-nofilter-lr0.000500-h2056-l1/weights/f1/model_fold1.pth"}, {"dataset": "temporal-sequential-conversation-v2-dataset-onehot-04-realtest"}),
    #         ("eval", {"path": '', "use_current_session": True}, dict()),
    #     ],
    #     "model_configs": {
    #         "activation": ("relu", dict()),
    #         # "loss_func": ("weighted-binary-cross-entropy", {"pos_weight": 2}),
    #         "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(2)}),
    #         # "loss_func": ("BCEW", {"reduction": "sum"}),
    #         "lr": 0.0005,
    #         'hidden_size': 2056,
    #         'num_layers': 1,
    #         "module_session_path": "output",
    #         "session_path_include_time": True,
    #     },
    # },
######################### Transformers Sessions
    # "bert": {
    #     "model": "bert-classifier",
    #     "commands": [
    #         ("train", {
    #             "epoch_num": 10,
    #             "batch_size": 32,
    #             "weights_checkpoint_paths": "",
    #             "condition_save_threshold": 0.05,
    #             },
    #             {
    #                 "dataset": "bert-based-uncased-tokenized",
    #                 "rerun_splitting": False,
    #                 "persist_splits": True,
    #                 "load_splits_from": None,
    #                 "n_splits": 5,
    #             }
    #         ),
    #         ("test", {"weights_checkpoint_paths": ""}, {"dataset": "bert-based-uncased-tokenized"}),
    #         # ("test", {"weights_checkpoint_paths": r"output/06-06-2023-16-17-29-lstm-balanced-v2-04-temporal/lstm/temporal-sequential/psw.rr.idr-v13000-nofilter-lr0.000500-h2056-l1/weights/f1/model_fold1.pth"}, {"dataset": "temporal-sequential-conversation-v2-dataset-onehot-04-realtest"}),
    #         ("eval", {"path": '', "use_current_session": True}, dict()),
    #     ],
    #     "model_configs": {
    #         "activation": ("relu", dict()),
    #         # "loss_func": ("weighted-binary-cross-entropy", {"pos_weight": 2}),
    #         "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(25)}),
    #         # "loss_func": ("BCEW", {"reduction": "sum"}),
    #         "lr": 0.0005,
    #         'hidden_size': 2056,
    #         'num_layers': 1,
    #         "module_session_path": "output",
    #         "session_path_include_time": True,
    #     },
    # },
    # "bert-toy": {
    #     "model": "bert-classifier",
    #     "commands": [
    #         ("train", {
    #             "epoch_num": 2,
    #             "batch_size": 8,
    #             "weights_checkpoint_paths": "",
    #             "condition_save_threshold": 0.05,
    #             },
    #             {
    #                 "dataset": "bert-based-uncased-tokenized-toy",
    #                 "rerun_splitting": False,
    #                 "persist_splits": True,
    #                 "load_splits_from": None,
    #                 "n_splits": 5,
    #             }
    #         ),
    #         ("test", {"weights_checkpoint_paths": ""}, {"dataset": "bert-based-uncased-tokenized-toy"}),
    #         # ("test", {"weights_checkpoint_paths": r"output/06-06-2023-16-17-29-lstm-balanced-v2-04-temporal/lstm/temporal-sequential/psw.rr.idr-v13000-nofilter-lr0.000500-h2056-l1/weights/f1/model_fold1.pth"}, {"dataset": "temporal-sequential-conversation-v2-dataset-onehot-04-realtest"}),
    #         ("eval", {"path": '', "use_current_session": True}, dict()),
    #     ],
    #     "model_configs": {
    #         "activation": ("relu", dict()),
    #         # "loss_func": ("weighted-binary-cross-entropy", {"pos_weight": 2}),
    #         "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(2.5)}),
    #         # "loss_func": ("BCEW", {"reduction": "sum"}),
    #         "lr": 0.0005,
    #         'hidden_size': 2056,
    #         'num_layers': 1,
    #         "module_session_path": "output",
    #         "session_path_include_time": True,
    #     },
    # },
#########################
    # "lstm-balanced-v2-04-temporal": {
    #     "model": "lstm",
    #     "commands": [
    #         # ("train", {
    #         #     "epoch_num": 20,
    #         #     "batch_size": 16,
    #         #     "weights_checkpoint_paths": "",
    #         #     },
    #         #     {
    #         #         "dataset": "temporal-sequential-conversation-v2-dataset-onehot-04",
    #         #         "rerun_splitting": False,
    #         #         "persist_splits": True,
    #         #         "load_splits_from": None,
    #         #         "n_splits": 5,
    #         #     }
    #         # ),
    #         # ("test", {"weights_checkpoint_paths": ""}, {"dataset": "temporal-sequential-conversation-v2-dataset-onehot-04"}),
    #         ("test", {"weights_checkpoint_paths": r"output/06-06-2023-16-17-29-lstm-balanced-v2-04-temporal/lstm/temporal-sequential/psw.rr.idr-v13000-nofilter-lr0.000500-h2056-l1/weights/f1/model_fold1.pth"}, {"dataset": "temporal-sequential-conversation-v2-dataset-onehot-04"}),
    #         ("eval", {"path": '', "use_current_session": True}, dict()),
    #     ],
    #     "model_configs": {
    #         "activation": ("relu", dict()),
    #         # "loss_func": ("weighted-binary-cross-entropy", {"pos_weight": 2}),
    #         "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(2)}),
    #         # "loss_func": ("BCEW", {"reduction": "sum"}),
    #         "lr": 0.0005,
    #         'hidden_size': 2056,
    #         'num_layers': 1,
    #         "module_session_path": "output",
    #         "session_path_include_time": True,
    #     },
    # },


    # "lstm-balanced-v2-02-temporal-realtest": {
    #     "model": "lstm",
    #     "commands": [
    #         ("train", {
    #             "epoch_num": 40,
    #             "batch_size": 16,
    #             "weights_checkpoint_paths": "",
    #             },
    #             {
    #                 "dataset": "temporal-sequential-conversation-v2-dataset-onehot-02-realtest",
    #                 "rerun_splitting": False,
    #                 "persist_splits": True,
    #                 "load_splits_from": None,
    #                 "n_splits": 5,
    #             }
    #         ),
    #         ("test", {"weights_checkpoint_paths": ""},
    #             {"dataset": "temporal-sequential-conversation-v2-dataset-onehot-02-realtest"}),
    #         ("eval", {"path": '', "use_current_session": True}, dict()),
    #     ],
    #     "model_configs": {
    #         "activation": ("relu", dict()),
    #         # "loss_func": ("weighted-binary-cross-entropy", {"pos_weight": 2}),
    #         "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(3.5)}),
    #         "lr": 0.009,
    #         'hidden_size': 1024,
    #         'num_layers': 1,
    #         "module_session_path": "output",
    #         "session_path_include_time": True,
    #     },
    # },





    "feedforward-toy": {
        "model": "ann",
        "commands": [
            ("train", {
                "epoch_num": 2,
                "batch_size": 32,
                "weights_checkpoint_path": "",
                },
                {
                    "dataset": "toy-conversation-v2-dataset-onehot",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": None,
                    "n_splits": 5,
                }
            ),
            ("test", dict(), {"dataset": "toy-conversation-v2-dataset-onehot"}),
            ("eval", {"path": '', "use_current_session": True}, dict()),
        ],
        "model_configs": {
            "dimension_list": list([8]),
            "dropout_list": [0.0],
            "activation": ("relu", dict()),
            "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(30)}),
            "lr": 0.005,
            "module_session_path": "output",
            "session_path_include_time": False,
        },
    },
}


USE_CUDA_IF_AVAILABLE = True

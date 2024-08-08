import torch

__preprocessings__ = [] ## Just to make it easier to change configurations

datasets = {
    "temporal-nauthor-sequential-conversation-distilroberta": (
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/train.csv",
            "output_path": "data/preprocessed/early-detection/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "",
            "output_path": "data/preprocessed/early-detection/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": False,
        }
    ),

    "temporal-nauthor-sequential-conversation-distilroberta-nllb-de-is-predatory": (
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory_nllb-deu_Latn-isl_Latn.csv",
            "output_path": "data/preprocessed/early-detection/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "",
            "output_path": "data/preprocessed/early-detection/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": False,
        }
    ),
    
    "temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-ca-predatory": (
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory_nllb-fra_Latn-cat_Latn.csv",
            "output_path": "data/preprocessed/early-detection/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "",
            "output_path": "data/preprocessed/early-detection/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": False,
        }
    ),
}

max_passed_messages = 10
sessions = {
    "lstm-distilroberta": {
        "model": "lstm",
        "commands": [
            ("train", {
                "epoch_num": 20,
                "batch_size": 8,
                "weights_checkpoint_path": "",
                "condition_save_threshold": 1000,
                "max_passed_messages": max_passed_messages,
                },
                {
                    "dataset": f"temporal-nauthor-sequential-conversation-distilroberta",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "",
                    "n_splits": 3,
                    "validate-on-test": False, 
                }
            ),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/lstm-distilroberta/lstm/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 10},  {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/lstm-distilroberta/lstm/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 20},  {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/lstm-distilroberta/lstm/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 30},  {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/lstm-distilroberta/lstm/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 50},  {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/lstm-distilroberta/lstm/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 75},  {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/lstm-distilroberta/lstm/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 100}, {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/lstm-distilroberta/lstm/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 150}, {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/lstm-distilroberta/lstm/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],},
                      {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),

            ("eval", {"path": f'output-earlydetection/lstm-distilroberta/lstm/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}',
                      "use_current_session": True}, {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
        ],
        "model_configs": {
            "activation": ("relu", dict()),
            "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(16.5)}),
            "lr": 0.0005,
            'hidden_size': 512,
            'num_layers': 1,
            "module_session_path": "output-earlydetection",
            "session_path_include_time": False,
            "early_stop": False,
        },
    },

    "gru-distilroberta": {
        "model": "gru",
        "commands": [
            ("train", {
                "epoch_num": 20,
                "batch_size": 8,
                "weights_checkpoint_path": "",
                "condition_save_threshold": 1000,
                "max_passed_messages": max_passed_messages,
                },
                {
                    "dataset": f"temporal-nauthor-sequential-conversation-distilroberta",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "",
                    "n_splits": 3,
                    "validate-on-test": False, 
                }
            ),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/gru-distilroberta/gru/temporal-nauthor-sequential-embedding/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 10},  {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/gru-distilroberta/gru/temporal-nauthor-sequential-embedding/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 20},  {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/gru-distilroberta/gru/temporal-nauthor-sequential-embedding/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 30},  {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/gru-distilroberta/gru/temporal-nauthor-sequential-embedding/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 50},  {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/gru-distilroberta/gru/temporal-nauthor-sequential-embedding/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 75},  {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/gru-distilroberta/gru/temporal-nauthor-sequential-embedding/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 100}, {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/gru-distilroberta/gru/temporal-nauthor-sequential-embedding/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 150}, {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/gru-distilroberta/gru/temporal-nauthor-sequential-embedding/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],},
                      {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),

            ("eval", {"path": f'output-earlydetection/gru-distilroberta/gru/temporal-nauthor-sequential-embedding/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}',
                      "use_current_session": True}, {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}), # the eval dataset, should be the same as training
        ],
        "model_configs": {
            "activation": ("relu", dict()),
            "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(16.5)}),
            "lr": 0.0005,
            'hidden_size': 512,
            'num_layers': 1,
            "module_session_path": "output-earlydetection",
            "session_path_include_time": False,
            "early_stop": False,
        },
    },
    ##################### NLLB de+is
    "lstm-distilroberta-1": {
        "model": "lstm",
        "commands": [
            ("train", {
                "epoch_num": 20,
                "batch_size": 8,
                "weights_checkpoint_path": "",
                "condition_save_threshold": 1000,
                "max_passed_messages": max_passed_messages,
                },
                {
                    "dataset": f"temporal-nauthor-sequential-conversation-distilroberta",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "",
                    "n_splits": 3,
                    "validate-on-test": False, 
                }
            ),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/lstm-distilroberta-1/lstm/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-de-is-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 10},  {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/lstm-distilroberta-1/lstm/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-de-is-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 20},  {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/lstm-distilroberta-1/lstm/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-de-is-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 30},  {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/lstm-distilroberta-1/lstm/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-de-is-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 50},  {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/lstm-distilroberta-1/lstm/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-de-is-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 75},  {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/lstm-distilroberta-1/lstm/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-de-is-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 100}, {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/lstm-distilroberta-1/lstm/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-de-is-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 150}, {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/lstm-distilroberta-1/lstm/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-de-is-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],},
                      {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),

            ("eval", {"path": f'output-earlydetection/lstm-distilroberta-1/lstm/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-de-is-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}',
                      "use_current_session": True}, {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
        ],
        "model_configs": {
            "activation": ("relu", dict()),
            "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(16.5)}),
            "lr": 0.0005,
            'hidden_size': 512,
            'num_layers': 1,
            "module_session_path": "output-earlydetection",
            "session_path_include_time": False,
            "early_stop": False,
        },
    },

    "gru-distilroberta-1": {
        "model": "gru",
        "commands": [
            ("train", {
                "epoch_num": 20,
                "batch_size": 8,
                "weights_checkpoint_path": "",
                "condition_save_threshold": 1000,
                "max_passed_messages": max_passed_messages,
                },
                {
                    "dataset": f"temporal-nauthor-sequential-conversation-distilroberta",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "",
                    "n_splits": 3,
                    "validate-on-test": False, 
                }
            ),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/gru-distilroberta-1/gru/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-de-is-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 10},  {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/gru-distilroberta-1/gru/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-de-is-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 20},  {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/gru-distilroberta-1/gru/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-de-is-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 30},  {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/gru-distilroberta-1/gru/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-de-is-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 50},  {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/gru-distilroberta-1/gru/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-de-is-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 75},  {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/gru-distilroberta-1/gru/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-de-is-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 100}, {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/gru-distilroberta-1/gru/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-de-is-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 150}, {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/gru-distilroberta-1/gru/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-de-is-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],},
                    {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),

            ("eval", {"path": f'output-earlydetection/gru-distilroberta-1/gru/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-de-is-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}',
                      "use_current_session": True}, {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}), # the eval dataset, should be the same as training
        ],
        "model_configs": {
            "activation": ("relu", dict()),
            "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(16.5)}),
            "lr": 0.0005,
            'hidden_size': 512,
            'num_layers': 1,
            "module_session_path": "output-earlydetection",
            "session_path_include_time": False,
            "early_stop": False,
        },
    },
    ##################

    "lstm-distilroberta-2": {
        "model": "lstm",
        "commands": [
            ("train", {
                "epoch_num": 20,
                "batch_size": 8,
                "weights_checkpoint_path": "",
                "condition_save_threshold": 1000,
                "max_passed_messages": max_passed_messages,
                },
                {
                    "dataset": f"temporal-nauthor-sequential-conversation-distilroberta",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "",
                    "n_splits": 3,
                    "validate-on-test": False, 
                }
            ),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/lstm-distilroberta-2/lstm/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-ca-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 10},  {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/lstm-distilroberta-2/lstm/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-ca-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 20},  {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/lstm-distilroberta-2/lstm/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-ca-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 30},  {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/lstm-distilroberta-2/lstm/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-ca-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 50},  {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/lstm-distilroberta-2/lstm/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-ca-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 75},  {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/lstm-distilroberta-2/lstm/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-ca-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 100}, {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/lstm-distilroberta-2/lstm/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-ca-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 150}, {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/lstm-distilroberta-2/lstm/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-ca-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],},
                    {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),

            ("eval", {"path": f'output-earlydetection/lstm-distilroberta-2/lstm/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-ca-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}',
                      "use_current_session": True}, {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
        ],
        "model_configs": {
            "activation": ("relu", dict()),
            "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(16.5)}),
            "lr": 0.0005,
            'hidden_size': 512,
            'num_layers': 1,
            "module_session_path": "output-earlydetection",
            "session_path_include_time": False,
            "early_stop": False,
        },
    },

    "gru-distilroberta-2": {
        "model": "gru",
        "commands": [
            ("train", {
                "epoch_num": 20,
                "batch_size": 8,
                "weights_checkpoint_path": "",
                "condition_save_threshold": 1000,
                "max_passed_messages": max_passed_messages,
                },
                {
                    "dataset": f"temporal-nauthor-sequential-conversation-distilroberta",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "",
                    "n_splits": 3,
                    "validate-on-test": False, 
                }
            ),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/gru-distilroberta-2/gru/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-ca-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 10},  {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/gru-distilroberta-2/gru/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-ca-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 20},  {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/gru-distilroberta-2/gru/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-ca-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 30},  {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/gru-distilroberta-2/gru/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-ca-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 50},  {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/gru-distilroberta-2/gru/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-ca-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 75},  {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/gru-distilroberta-2/gru/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-ca-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 100}, {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/gru-distilroberta-2/gru/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-ca-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],
                "max_passed_messages": 150}, {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),
            ("test", {"regex_weights_checkpoint_path": [f"output-earlydetection/gru-distilroberta-2/gru/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-ca-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}/**/model_f*.pth"],},
                    {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}),

            ("eval", {"path": f'output-earlydetection/gru-distilroberta-2/gru/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-ca-predatory/p-v768-filtered-lr0.000500-h512-l1/n{max_passed_messages}',
                      "use_current_session": True}, {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta"}), # the eval dataset, should be the same as training
        ],
        "model_configs": {
            "activation": ("relu", dict()),
            "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(16.5)}),
            "lr": 0.0005,
            'hidden_size': 512,
            'num_layers': 1,
            "module_session_path": "output-earlydetection",
            "session_path_include_time": False,
            "early_stop": False,
        },
    },
}


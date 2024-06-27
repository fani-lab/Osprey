import torch

USE_CUDA_IF_AVAILABLE = True
__preprocessings__ = [] ## Just to make it easier to change configurations
AGGERAGETD_METRICS_PATH = "agg-rtt.csv"

datasets = {

    "temporal-nauthor-sequential-conversation-distilroberta-nllb-en-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory_nllb-eng_Latn.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
            
        }
    ),

    "temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory_nllb-fra_Latn.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
            
        }
    ),

    "temporal-nauthor-sequential-conversation-distilroberta-nllb-de-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory_nllb-deu_Latn.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
            
        }
    ),

    "temporal-nauthor-sequential-conversation-distilroberta-nllb-is-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory_nllb-isl_Latn.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
            
        }
    ),

    "temporal-nauthor-sequential-conversation-distilroberta-nllb-cat-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory_nllb-cat_Latn.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),

    "temporal-nauthor-sequential-conversation-distilroberta-nllb-pes-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory_nllb-pes_Arab.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),

    "temporal-nauthor-sequential-conversation-distilroberta-nllb-hin-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory_nllb-hin_Deva.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),

    "temporal-nauthor-sequential-conversation-distilroberta-nllb-zho-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory_nllb-zho_Hans.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),

    "temporal-nauthor-sequential-conversation-distilroberta-nllb-mya-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory_nllb-mya_Mymr.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),

    "temporal-nauthor-sequential-conversation-distilroberta-nllb-ps-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory_nllb-pbt_Arab.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),

    # bcewithlogits --> 3.25
    "temporal-nauthor-sequential-conversation-distilroberta-nllb-pes-fra-zho-deu-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory_nllb-pes_Arab-fra_Latn-zho_Hans-deu_Latn.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),

    # bcewithlogits --> 3.25
    "temporal-nauthor-sequential-conversation-distilroberta-nllb-is-hi-ca-my-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory_nllb-isl_Latn-hin_Deva-cat_Latn-mya_Mymr.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),

    "temporal-nauthor-sequential-conversation-distilroberta-nllb-ca-ps-is-my-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory_nllb-cat_Latn-pbt_Arab-isl_Latn-mya_Mymr.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),

    # bcewithlogits --> 1.82
    "temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-fa-de-zh-is-hi-ca-my-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory_nllb-pes_Arab-fra_Latn-zho_Hans-deu_Latn-isl_Latn-hin_Deva-cat_Latn-mya_Mymr.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),


    ################## Backtranslation - m2m100-1.2B-f16
    "temporal-nauthor-sequential-conversation-distilroberta-m2m100-1.2B-f16-en-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory-m2m100-1.2B-f16-en_noformat.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),

    "temporal-nauthor-sequential-conversation-distilroberta-m2m100-1.2B-f16-fr-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory-m2m100-1.2B-f16-fr_noformat.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),

    "temporal-nauthor-sequential-conversation-distilroberta-m2m100-1.2B-f16-fa-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory-m2m100-1.2B-f16-fa_noformat.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),

    "temporal-nauthor-sequential-conversation-distilroberta-m2m100-1.2B-f16-zh-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory-m2m100-1.2B-f16-zh_noformat.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),

    "temporal-nauthor-sequential-conversation-distilroberta-m2m100-1.2B-f16-de-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory-m2m100-1.2B-f16-zh_noformat.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),

    "temporal-nauthor-sequential-conversation-distilroberta-m2m100-1.2B-f16-ca-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory-m2m100-1.2B-f16-ca_noformat.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),

    "temporal-nauthor-sequential-conversation-distilroberta-m2m100-1.2B-f16-hi-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory-m2m100-1.2B-f16-hi_noformat.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),

    "temporal-nauthor-sequential-conversation-distilroberta-m2m100-1.2B-f16-is-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory-m2m100-1.2B-f16-is_noformat.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),

    "temporal-nauthor-sequential-conversation-distilroberta-m2m100-1.2B-f16-my-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory-m2m100-1.2B-f16-my_noformat.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),

    "temporal-nauthor-sequential-conversation-distilroberta-m2m100-1.2B-f16-ps-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory-m2m100-1.2B-f16-ps_noformat.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),

    # bcewithlogits --> 3.25
    "temporal-nauthor-sequential-conversation-distilroberta-m2m100-1.2B-f16-fr-fa-zh-de-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory-m2m100-1.2B-f16-fr-fa-zh-de.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),

    # bcewithlogits --> 3.25
    "temporal-nauthor-sequential-conversation-distilroberta-m2m100-1.2B-f16-is-hi-ca-my-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory-m2m100-1.2B-f16-is-hi-ca-my.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),

    # bcewithlogits --> 3.25
    "temporal-nauthor-sequential-conversation-distilroberta-m2m100-1.2B-f16-ca-ps-is-my-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory-m2m100-1.2B-f16-ca-ps-is-my.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),

    # bcewithlogits --> 1.82
    "temporal-nauthor-sequential-conversation-distilroberta-m2m100-1.2B-f16-fr-fa-de-zh-is-hi-ca-my-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory-m2m100-1.2B-f16-fr-fa-de-zh-is-hi-ca-my.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),

    ############################ Backtranslations from Google Cloud Platform Translation API
    "temporal-nauthor-sequential-conversation-distilroberta-gcpt-fr-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory-gcpt-fr.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),

    "temporal-nauthor-sequential-conversation-distilroberta-gcpt-fa-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory-gcpt-fa.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),

    "temporal-nauthor-sequential-conversation-distilroberta-gcpt-de-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory-gcpt-de.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),

    "temporal-nauthor-sequential-conversation-distilroberta-gcpt-zh-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory-gcpt-zh.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),

    "temporal-nauthor-sequential-conversation-distilroberta-gcpt-is-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory-gcpt-is.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),

    "temporal-nauthor-sequential-conversation-distilroberta-gcpt-hi-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory-gcpt-hi.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),

    "temporal-nauthor-sequential-conversation-distilroberta-gcpt-ca-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory-gcpt-ca.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),

    "temporal-nauthor-sequential-conversation-distilroberta-gcpt-my-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory-gcpt-my.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),


    "temporal-nauthor-sequential-conversation-distilroberta-gcpt-fr-fa-de-zh-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory-gcpt-fr-fa-de-zh.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),

    "temporal-nauthor-sequential-conversation-distilroberta-gcpt-is-hi-ca-my-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory-gcpt-ca-hi-is-my.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),

    # bcewithlogits --> 1.82
    "temporal-nauthor-sequential-conversation-distilroberta-gcpt-fr-fa-de-zh-ca-hi-is-my-predators": ( # the translations are only predatory messages
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/translated/predatory-gcpt-fr-fa-de-zh-ca-hi-is-my.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": True,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),

    #################### No backtranslations, all training at once
    "temporal-nauthor-sequential-conversation-distilroberta": (
        "temporal-nauthor-sequential-embedding",  # short name of the dataset
        {       # train configs
            "data_path": "data/dataset-v2/train.csv",
            "output_path": "data/preprocessed/sequential-v2/",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        },
        {      # test configs
            "data_path": "data/dataset-v2/test.csv",
            "forced_output_path": "data/preprocessed/sequential-v2/test-temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-nllb-fr-predators/p-v768-filtered",
            "output_path": "data/preprocessed/sequential-v2/test-",
            "load_from_pkl": True,
            "preprocessings": __preprocessings__,
            "persist_data": False,
            "apply_record_filter": True,
        }
    ),

    ###################################### End of backtranslation

}

# language_gru_0  = "m2m100-1.2B-f16-ps"
language_lstm_0 = "m2m100-1.2B-f16-ps"

language_gru_0  = "nllb-ps"
# language_lstm_0 = "nllb-ps"

# language_gru_0  = "gcpt-fr"
# language_lstm_0 = "gcpt-fr"


# language_gru_1  = "m2m100-1.2B-f16-fr-fa-de-zh-is-hi-ca-my"
# language_lstm_1 = "m2m100-1.2B-f16-fr-fa-de-zh-is-hi-ca-my"

# language_gru_1  = "nllb-fr-fa-de-zh-is-hi-ca-my"
# language_lstm_1 = "nllb-fr-fa-de-zh-is-hi-ca-my"

# language_gru_1  = "gcpt-fr-fa-de-zh-ca-hi-is-my"
# language_lstm_1 = "gcpt-fr-fa-de-zh-ca-hi-is-my"

# language_gru_1  = "m2m100-1.2B-f16-ca-ps-is-my"
language_lstm_1 = "m2m100-1.2B-f16-ca-ps-is-my"

language_gru_1  = "nllb-ca-ps-is-my"
# language_lstm_1 = "nllb-ca-ps-is-my"



sessions = {
############## Distilroberta single validation
    "lstm-distilroberta-noval": {
        "model": "lstm",
        "commands": [
            ("train", {
                "epoch_num": 100,
                "batch_size": 8,
                "weights_checkpoint_path": "",
                "condition_save_threshold": 10000,
                },
                { # temporal-nauthor-sequential-conversation-v2-dataset-distilroberta, temporal-nauthor-sequential-conversation-dataset-distilroberta-pretrained
                  # sequential-conversation-dataset-distilroberta-pretrained, temporal-nauthor-sequential-conversation-distilroberta-en-fr, sequential-conversation-distilroberta-en-fr
                  # temporal-nauthor-sequential-conversation-distilroberta-en-fr-predators, toy-temporal-sequential-conversation-v2-dataset-distilroberta
                    "dataset": "temporal-nauthor-sequential-conversation-distilroberta-m2m100-1.2B-f16-en-predators",
                    "validate-on-test": True,
                },
            ),
        ],
        "model_configs": {
            "activation": ("relu", dict()),
            "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(8.1)}),
            "lr": 0.0005,
            'hidden_size': 514,
            'num_layers': 1,
            "module_session_path": "output-ecir2024",
            "session_path_include_time": False,
            "early_stop": False,
        },
    },

    "gru-distilroberta-noval": {
        "model": "gru",
        "commands": [
            ("train", {
                "epoch_num": 100,
                "batch_size": 8,
                "weights_checkpoint_path": "",
                "condition_save_threshold": 10000,
                },
                {
                    "dataset": "temporal-nauthor-sequential-conversation-distilroberta-nllb-en-predators",
                    "validate-on-test": True,
                },
            ),
        ],
        "model_configs": {
            "activation": ("relu", dict()),
            "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(8.1)}),
            "lr": 0.0005,
            'hidden_size': 514,
            'num_layers': 1,
            "module_session_path": "output-ecir2024",
            "session_path_include_time": False,
            "early_stop": False,
        },
    },

######## With splits and everything
    "lstm-distilroberta-0": {
        "model": "lstm",
        "commands": [
            ("train", {
                "epoch_num": 20,
                "batch_size": 8,
                "weights_checkpoint_path": "",
                "condition_save_threshold": 1000,
                },
                {
                    "dataset": f"temporal-nauthor-sequential-conversation-distilroberta-{language_lstm_0}-predators",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "data/preprocessed/sequential-v2/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-m2m100-1.2B-f16-fr-predators/p-v768-filtered/splits-n3stratified.pkl",
                    "n_splits": 3,
                    "validate-on-test": False, 
                }
            ),
            ("test", {"weights_checkpoint_path": []}, {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta-{language_lstm_0}-predators"}),
            ("eval", {"path": '', "use_current_session": True}, {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta-{language_lstm_0}-predators"}),
        ],
        "model_configs": {
            "activation": ("relu", dict()),
            "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(8.1)}),
            "lr": 0.0005,
            'hidden_size': 512,
            'num_layers': 1,
            "module_session_path": "output-2024",
            "session_path_include_time": False,
            "early_stop": False,
        },
    },

    "gru-distilroberta-0": {
        "model": "gru",
        "commands": [
            ("train", {
                "epoch_num": 20,
                "batch_size": 8,
                "weights_checkpoint_path": "",
                "condition_save_threshold": 1000,
                },
                {
                    "dataset": f"temporal-nauthor-sequential-conversation-distilroberta-{language_gru_0}-predators",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "data/preprocessed/sequential-v2/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-m2m100-1.2B-f16-fr-predators/p-v768-filtered/splits-n3stratified.pkl",
                    "n_splits": 3,
                    "validate-on-test": False, 
                }
            ),
            ("test", {"weights_checkpoint_path": []}, {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta-{language_gru_0}-predators"}),
            ("eval", {"path": '', "use_current_session": True}, {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta-{language_gru_0}-predators"}),
        ],
        "model_configs": {
            "activation": ("relu", dict()),
            "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(8.1)}),
            "lr": 0.0005,
            'hidden_size': 512,
            'num_layers': 1,
            "module_session_path": "output-2024",
            "session_path_include_time": False,
            "early_stop": False,
        },
    },

    "lstm-distilroberta-1": {
        "model": "lstm",
        "commands": [
            ("train", {
                "epoch_num": 20,
                "batch_size": 8,
                "weights_checkpoint_path": "",
                },
                {
                    "dataset": f"temporal-nauthor-sequential-conversation-distilroberta-{language_lstm_1}-predators",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "data/preprocessed/sequential-v2/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-m2m100-1.2B-f16-fr-fa-zh-de-predators/p-v768-filtered/splits-n3stratified.pkl",
                    "n_splits": 3,
                    "validate-on-test": False, 
                }
            ),
            ("test", {"weights_checkpoint_path": []}, {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta-{language_lstm_1}-predators"}),
            ("eval", {"path": '', "use_current_session": True}, {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta-{language_lstm_1}-predators"}),
        ],
        "model_configs": {
            "activation": ("relu", dict()),
            "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(3.25)}),
            "lr": 0.0005,
            'hidden_size': 512,
            'num_layers': 1,
            "module_session_path": "output-2024",
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
                },
                {
                    "dataset": f"temporal-nauthor-sequential-conversation-distilroberta-{language_gru_1}-predators",
                    "rerun_splitting": False,
                    "persist_splits": True,
                    "load_splits_from": "data/preprocessed/sequential-v2/temporal-nauthor-sequential-embedding/temporal-nauthor-sequential-conversation-distilroberta-m2m100-1.2B-f16-fr-fa-zh-de-predators/p-v768-filtered/splits-n3stratified.pkl",
                    "n_splits": 3,
                    "validate-on-test": False, 
                }
            ),
            ("test", {"weights_checkpoint_path": []}, {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta-{language_gru_1}-predators"}),
            ("eval", {"path": '', "use_current_session": True}, {"dataset": f"temporal-nauthor-sequential-conversation-distilroberta-{language_gru_1}-predators"}),
        ],
        "model_configs": {
            "activation": ("relu", dict()),
            "loss_func": ("BCEW", {"reduction": "sum", "pos_weight": torch.tensor(3.25)}),
            "lr": 0.0005,
            'hidden_size': 512,
            'num_layers': 1,
            "module_session_path": "output-2024",
            "session_path_include_time": False,
            "early_stop": False,
        },
    },

}


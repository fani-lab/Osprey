import time
import logging
import sys

import torch.nn

from settings import settings, mappings

START_TIME = time.strftime("%m-%d-%Y-%H-%M-%S", time.localtime())

FORMATTER = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s : %(message)s")
FORMATTER_VERBOSE = logging.Formatter(
    "%(asctime)s | %(name)s | %(levelname)s | %(filename)s %(funcName)s @ %(lineno)s : %(message)s")

debug_file_handler = logging.FileHandler(f"logs/{START_TIME}.log")
debug_file_handler.setLevel(logging.DEBUG)
debug_file_handler.setFormatter(FORMATTER_VERBOSE)

info_terminal_handler = logging.StreamHandler(sys.stdout)
info_terminal_handler.setLevel(logging.INFO)
info_terminal_handler.setFormatter(FORMATTER)

logger = logging.getLogger()
logger.addHandler(debug_file_handler)
logger.addHandler(info_terminal_handler)
logger.setLevel(logging.DEBUG)


def create_model_configs(session_name: str, session: dict, device: str):
    activation, activation_kwargs = mappings.ACTIVATIONS[session["model_configs"]["activation"][0]], session["model_configs"]["activation"][1]
    loss, loss_kwargs = mappings.LOSS_FUNCTIONS[session["model_configs"]["loss_func"][0]], session["model_configs"]["loss_func"][1]

    configs = {**session["model_configs"], "activation": activation(**activation_kwargs),
                         "loss_func": loss(**loss_kwargs), "device": device,
                         "module_session_path": session["model_configs"]["module_session_path"] + "/" + START_TIME + "/" + session_name
                            if session["model_configs"]["session_path_include_time"] else session["model_configs"]["module_session_path"] + "/" + session_name,
                        }
    configs = {k: v for k, v in configs.items() if k not in settings.FILTERED_CONFIGS}
    return configs

def run():
    device = 'cuda' if settings.USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else 'cpu'
    logger.info(f'processing unit: {device}')
    datasets = dict()
    for dataset_name, (short_name, train_configs, test_configs) in settings.datasets.items():
        dataset_class = None
        try:
            dataset_class = mappings.DATASETS[short_name]
        except Exception as e:
            raise Exception(f"the dataset {short_name} is either not implemented or not registered")
                
        preprocessings = []
        for pp in train_configs["preprocessings"]:
            try:
                preprocessings.append(mappings.PREPROCESSINGS[pp])
            except Exception as e:
                raise Exception(f"preprocessing `{pp}` either not implemented or not registered") from e
        
        train_dataset = dataset_class(**{**train_configs, "preprocessings": [pp() for pp in preprocessings], "device": device})
        test_dataset = dataset_class(**{**test_configs, "parent_dataset": train_dataset, "preprocessings": [pp() for pp in preprocessings], "device": device})
        datasets[dataset_name] = (train_dataset, test_dataset)

    for model_name, session in settings.sessions.items():
        logger.info(f"started new session: {model_name}")
        commands = session["commands"]

        model_configs = create_model_configs(model_name, session=session, device=device)
        
        model_class = mappings.MODELS[session["model"]]
        
        for command, command_kwargs, dataset_name, *_ in commands:
            logger.info(f"started new command `{command}` of session `{model_name}`")
            logger.debug(f"command `{command}`; dataset name: {dataset_name}; arguments: {command_kwargs}")
            if command == "train":
                dataset = datasets[dataset_name][0]
                dataset.prepare()
                model = model_class(**model_configs, input_size=datasets[dataset_name][0].shape[1])
                model.to(device=device)
                model.learn(**command_kwargs, train_dataset=dataset)
            if command == "test":
                dataset = datasets[dataset_name][1]
                dataset.prepare()
                model = model_class(**model_configs, input_size=datasets[dataset_name][0].shape[1])
                model.to(device=device)
                model.test(**command_kwargs, test_dataset=dataset)
            if command == "eval":
                path = command_kwargs.get("path", "")
                if command_kwargs.get("use_current_session", False):
                    try:
                        path = model.get_detailed_session_path(dataset)
                    except UnboundLocalError as e:
                        raise Exception("in order to use use_current_session, you should run the previous steps at the same time.") from e
                if path == "":
                    raise ValueError("the given path is empty. It should point to the directory of model objects.")
                model = model_class(**model_configs, input_size=1)
                model.eval(path, device=device)

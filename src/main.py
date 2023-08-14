import logging

import torch

import settings
from src import mappings
from src.utils.commons import CommandObject

logger = logging.getLogger()

def create_model_configs(session_name: str, session: dict, device: str):
    activation, activation_kwargs = mappings.ACTIVATIONS[session["model_configs"]["activation"][0]], session["model_configs"]["activation"][1]
    loss, loss_kwargs = mappings.LOSS_FUNCTIONS[session["model_configs"]["loss_func"][0]], session["model_configs"]["loss_func"][1]
    logger.info(f"activation module kwargs: {activation_kwargs}")
    logger.info(f"loss module kwargs: {loss_kwargs}")
    configs = {**session["model_configs"], "activation": activation(**activation_kwargs),
                         "loss_func": loss(**loss_kwargs), "device": device,
                         "module_session_path": session["model_configs"]["module_session_path"] + "/" + f"{settings.get_start_time()}-{session_name}"
                            if session["model_configs"]["session_path_include_time"] else session["model_configs"]["module_session_path"] + "/" + session_name
                        }
    configs = {k: v for k, v in configs.items() if k not in settings.ALL_FILTERED_CONFIGS}
    return configs

def initiate_datasets(datasets_maps, device):
    datasets = dict()
    if not datasets_maps:
        return datasets
    for dataset_name, (short_name, train_configs, test_configs) in datasets_maps.items():
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
        test_dataset = dataset_class(**{**test_configs, "parent_dataset": train_dataset, "preprocessings": [pp() for pp in preprocessings], "device": device, "apply_record_filter": False})
        logger.info(f"train dataset `{dataset_name}`, shortname: `{short_name}` kwargs -> {train_configs}")
        logger.info(f"test dataset `{dataset_name}`, shortname: `{short_name}` kwargs -> {test_configs}")
        datasets[dataset_name] = (train_dataset, test_dataset)
    
    return datasets

class RunTrainPipeline(CommandObject):

    def get_actions_and_args(self):
        
        def run(sessions):
            device = 'cuda' if settings.USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else 'cpu'
            torch.autograd.set_detect_anomaly(True)
            logger.info(f'processing unit: {device}')
            datasets = initiate_datasets(settings.datasets, device)
            
            if isinstance(sessions, list) and len(sessions) > 0:
                sessions = [(name, settings.sessions[name]) for name in sessions]
            else:
                sessions = settings.sessions.items()
            
            for model_name, session in sessions:
                logger.info(f"started new session: {model_name}")
                commands = session["commands"]

                model_configs = create_model_configs(model_name, session=session, device=device)
                
                model_class = mappings.MODELS[session["model"]]

                for command, command_kwargs, dataset_configs, *_ in commands:
                    dataset_name = dataset_configs.get("dataset", None)
                    if dataset_name is None:
                        logger.warning("no dataset was specified.")
                    logger.info(f"started new command `{command}` of session `{model_name}`")
                    logger.debug(f"command `{command}`; dataset name: {dataset_name}; arguments: {command_kwargs}")
                    torch.cuda.empty_cache()
                    if command == "train":
                        dataset = datasets[dataset_name][0]
                        dataset.prepare()
                        logger.info(f"dataset short-name: {str(dataset)}")
                        split_again = dataset_configs.get("rerun_splitting", False)
                        n_splits = dataset_configs.get("n_splits")
                        persist_splits = dataset_configs.get("persist_splits", True)
                        persist_splits = dataset_configs.get("persist_splits", True)
                        load_splits_from = dataset_configs.get("load_splits_from", True)
                        splits = dataset.split_dataset_by_label(n_splits, split_again, persist_splits, True, load_splits_from)

                        model = model_class(**model_configs, input_size=dataset.shape[-1])
                        model.to(device=device)
                        model.learn(**command_kwargs, train_dataset=dataset, splits=splits)
                    if command == "test":
                        dataset = datasets[dataset_name][1]
                        dataset.prepare()
                        logger.info(f"dataset short-name: {str(dataset)}")
                        model = model_class(**model_configs, input_size=datasets[dataset_name][0].shape[-1])
                        model.to(device=device)
                        if not command_kwargs.get("weights_checkpoint_path", None):
                            command_kwargs["weights_checkpoint_path"] = model.get_all_folds_checkpoints(datasets[dataset_name][0])
                        if isinstance(command_kwargs["weights_checkpoint_path"], str):
                            command_kwargs["weights_checkpoint_path"] = (command_kwargs["weights_checkpoint_path"],)
                        model.test(**command_kwargs, test_dataset=dataset)
                    if command == "eval":
                        path = command_kwargs.get("path", "")
                        if command_kwargs.get("use_current_session", False):
                            try:
                                datasets[dataset_name][0].prepare()
                                path = model.get_detailed_session_path(datasets[dataset_name][0])
                            except UnboundLocalError as e:
                                raise Exception("in order to use use_current_session, you should run the previous steps at the same time.") from e
                        if path == "":
                            raise ValueError("the given path is empty. It should point to the directory of model objects.")
                        model = model_class(**model_configs, input_size=1)
                        model.evaluate(path, device=device)

        return (run, [{
                "flags": ("-s", "--sessions"),
                "dest": "sessions",
                "nargs": "*",
                "default": [],
                "help": "the name of sessions to be run, as specified in settings file",
            }
        ])
    
    @classmethod
    def command(cls) -> str:
        return "train"
    
    def help(self) -> str:
        return "runs the training pipline for models specified in the settings file"
    
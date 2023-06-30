from src.preprocessing import BasePreprocessing
from src.utils.dataset import BaseDataset
from src.models.baseline import Baseline
from src.utils.commons import RegisterableObject, CommandObject
from src.utils.loss_functions import BaseLossCalculator

from torch.nn import ReLU, CrossEntropyLoss, BCEWithLogitsLoss, BCELoss

MODELS = dict()

PREPROCESSINGS = dict()

DATASETS = dict()

ACTIVATIONS = dict()

LOSS_FUNCTIONS = dict()

COMMANDS = dict()

def register_mappings(obj: RegisterableObject):
    
    if issubclass(obj, BasePreprocessing):
        if PREPROCESSINGS.get(obj.short_name(), None) is not None:
            raise Exception(f"a class of the same shortname `{obj.short_name()}` already registered")
        PREPROCESSINGS[obj.short_name()] = obj
    
    if issubclass(obj, Baseline):
        if PREPROCESSINGS.get(obj.short_name(), None) is not None:
            raise Exception(f"a class of the same shortname `{obj.short_name()}` already registered")
        MODELS[obj.short_name()] = obj
        
    if issubclass(obj, BaseDataset):
        if DATASETS.get(obj.short_name(), None) is not None:
            raise Exception(f"a class of the same shortname `{obj.short_name()}` already registered")
        DATASETS[obj.short_name()] = obj
    
    if issubclass(obj, BaseLossCalculator):
        if LOSS_FUNCTIONS.get(obj.short_name(), None) is not None:
            raise Exception(f"a class of the same shortname `{obj.short_name()}` already registered")
        LOSS_FUNCTIONS[obj.short_name()] = obj

# it shouldn't be really like this, but to override torch classes and make them inherit RegisterableObject
def register_mappings_torch():
    ACTIVATIONS["relu"] = ReLU
    LOSS_FUNCTIONS["cross-entropy"] = CrossEntropyLoss
    LOSS_FUNCTIONS["BCEW"] = BCEWithLogitsLoss
    LOSS_FUNCTIONS["BCELoss"] = BCELoss

    pass

def register_command(obj: CommandObject):
    
    if not issubclass(obj, CommandObject):
        raise ValueError(f"the following class cannot be registered as a command.")
    
    if COMMANDS.get(obj.command(), None) is not None:
        raise ValueError(f"the command `{obj.command()}` already exists.")
    COMMANDS[obj.command()] = obj
    
from src.utils.commons import CommandObject
from src.mappings import PREPROCESSINGS, ACTIVATIONS, DATASETS, MODELS, LOSS_FUNCTIONS

TABS = 4
NEST_TAB = 4

def __print_mapping__(mapping, name):
    if len(mapping) == 0:
        return f"nothing registered for {name}"
    result = f"{name}:\n"
    max_key_length = 0
    for k in mapping.keys():
        max_key_length = max_key_length if max_key_length > len(k) else len(k)
    
    for k, v in mapping.items():
        l = max_key_length - len(k) + TABS
        result += f"{'':>{NEST_TAB}}{k}{'':>{l}}{v.__name__}\n"
    
    print(result)

class PrintMappings(CommandObject):

    def get_actions_and_args(self):
        
        def callback():
            __print_mapping__(PREPROCESSINGS, "preprocessings")
            __print_mapping__(ACTIVATIONS, "activations")
            __print_mapping__(DATASETS, "datasets")
            __print_mapping__(MODELS, "models")
            __print_mapping__(LOSS_FUNCTIONS, "loss functions")
        
        return callback, []
    
    @classmethod
    def command(cls) -> str:
        return "mappings"
    
    def help(self) -> str:
        return "prints the mappings from string representation to modules and objects. The string representation can be used in modifying settings file"

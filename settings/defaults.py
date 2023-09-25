try:
    from .settings import USE_CUDA_IF_AVAILABLE
except:
    USE_CUDA_IF_AVAILABLE = False
try:
    from .settings import IGNORED_PARAM_RESET
except:
    IGNORED_PARAM_RESET = set()

try:
    from .settings import FILTERED_CONFIGS
except:
    FILTERED_CONFIGS = set()
TRAIN = 1
TEST  = 2
EVAL  = 4
__start_time__ = 0

ALL_FILTERED_CONFIGS = {
    "session_path_include_time",
    "data_path",
    "output_path",
    "load_from_pkl",
    "preprocessings",
    "persist_data",
    "splitting_configs",
} or FILTERED_CONFIGS

ALL_IGNORED_PARAM_RESET = {"activation", "loss_function"} or IGNORED_PARAM_RESET
OUTPUT_LAYER_NODES = 1

def get_start_time():
    global __start_time__
    if __start_time__ == 0:
        import time
        __start_time__ = time.strftime("%m-%d-%Y-%H-%M-%S", time.localtime())
        del time
    return __start_time__

AGGERAGETD_METRICS_PATH = "agg.csv"

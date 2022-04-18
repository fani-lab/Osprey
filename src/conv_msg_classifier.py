from baseline import Baseline
from msg_classifier import msg_classifier

class conv_msg_classifier(msg_classifier):
    def __init__(self):
        super(msg_classifier, self).__init__()
        conversation_feature_sets = [['line'], ['time'], ['n_msgs'], ['is_binconv']]
        pass
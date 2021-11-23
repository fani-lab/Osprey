from baseline import Baseline
import msg_classifier
class conv_msg_classifier(msg_classifier):
    conversation_feature_sets = [['line'], ['time'], ['n_msgs'], ['is_binconv']]
    pass
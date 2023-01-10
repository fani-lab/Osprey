from baseline import Baseline


class msg_classifier(Baseline):
    def __init__(self, features, split=[70, 30], target=['tagged_msg']):
        super(msg_classifier, self).__init__(features, target, split)


class conv_msg_classifier(msg_classifier):
    def __init__(self):
        super(msg_classifier, self).__init__()
        conversation_feature_sets = [['line'], ['time'], ['n_msgs'], ['is_binconv']]
        pass

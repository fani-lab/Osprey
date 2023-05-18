import pickle
import logging
import matplotlib.pyplot as plt
from src.utils.commons import RegisterableObject, roc_auc, calculate_metrics, roc, precision_recall_auc, precision_recall_curve


logger = logging.getLogger()


class Baseline(RegisterableObject):

    def __init__(self, input_size: int, activation, loss_func, lr, module_session_path, validation_steps=-1,
                 device='cpu', **kwargs):
        super().__init__()
        self.input_size = input_size
        self.init_lr = lr
        self.validation_steps = validation_steps
        self.activation = activation

        self.loss_function = loss_func

        self.session_path = module_session_path if module_session_path[-1] == "\\" or module_session_path[
            -1] == "/" else module_session_path + "/"

        self.snapshot_steps = 2
        self.device = device

    def learn(self):
        raise NotImplementedError()

    def test(self):
        # load the saved model and apply it on test set
        # save the prediction results
        raise NotImplementedError()

    def get_session_path(self, *args):
        raise NotImplementedError()
    
    def get_new_optimizer(self, lr, *args, **kwargs):
        raise NotImplementedError()
    
    def get_new_scheduler(self, optimizer, *args, **kwargs):
        raise NotImplementedError()

    def evaluate(self, path, device):

        preds = None
        targets = None
        with open(path+'/preds.pkl', 'rb') as file:
            preds = pickle.load(file)
        with open(path+'/targets.pkl', 'rb') as file:
            targets = pickle.load(file)
            targets = targets.to(device)
        if preds.ndim > targets.ndim:
            preds = preds.squeeze()
        preds = preds.to(device)
        # targets = torch.argmax(targets, dim=1)
        fpr, tpr, _ = roc(preds, targets, device=device)
        auroc = roc_auc(preds, targets, device=device)

        roc_path = path + "/ROC-curve.png"
        precision_recall_path = path + "/precision-recall-curve.png"
        plt.clf()
        plt.plot(fpr.cpu(), tpr.cpu())
        plt.title("ROC")
        plt.savefig(roc_path)
        logger.info(f"saving ROC curve at: {roc_path}")
        precisions, recalls, _ = precision_recall_curve(preds, targets)
        pr_auc = precision_recall_auc(preds, targets, device=device)
        plt.clf()
        plt.plot(recalls.cpu(), precisions.cpu())
        plt.title("Recall-Precision Curve")
        plt.savefig(precision_recall_path)
        logger.info(f"saving precision-recall curve at: {precision_recall_path}")
        # plt.show()
        accuracy, precision, recall = calculate_metrics(preds, targets, device=device)
        logger.info(f"test set -> AUCROC: {(auroc):>0.7f} | AUCPR: {(pr_auc):>0.7f} | accuracy: {(accuracy):>0.7f} | precision: {(precision):>0.7f} | recall: {(recall):>0.7f}")

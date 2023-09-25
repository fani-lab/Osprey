import pickle
import logging
from glob import glob
import re

from src.models.baseline import Baseline
from src.utils.commons import force_open, calculate_metrics_extended
import settings

from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger()

class BaseSingleVectorMachine(Baseline):

    def __init__(self, *args, **kwargs):
        Baseline.__init__(self, *args, **kwargs)
        self.device = "cpu"
        self.svm = SVC(kernel="rbf", )

    def to(self, *args, **kwargs):
        logger.warning("this model only runs on CPU")

    @classmethod
    def short_name(cls) -> str:
        return "base-svm"

    def learn(self, splits: list, train_dataset: Dataset, weights_checkpoint_path=None):
        if weights_checkpoint_path is not None and len(weights_checkpoint_path) > 0:
            self.load_params(weights_checkpoint_path)
        
        for fold, (train_ids, validation_ids) in enumerate(splits):
            logger.info(f'fetching data for fold #{fold}')
            train_loader, validation_loader = self.get_dataloaders(train_dataset, train_ids, validation_ids, -1)

            logger.info("fitting svm model")
            X, y = next(iter(train_loader))
            X = X.to_dense().cpu()
            y = y.cpu().squeeze() # if number of classes in y goes over one, remove squeeze()
            self.svm.fit(X, y)
            logger.info("validating svm model")
            X, y = next(iter(validation_loader))
            X = X.to_dense().cpu()
            y = y.cpu().squeeze()
            preds = self.svm.predict(X)
            preds = torch.tensor(preds)
            accuracy_value, recall_value, precision_value, f2score, f05score = calculate_metrics_extended(preds, y, device=self.device)
            logger.info(f"fold: {fold} | validation -> accuracy: {(100 * accuracy_value):>0.6f} | precision: {(100 * precision_value):>0.6f} | recall: {(100 * recall_value):>0.6f} | f2: {(100 * f2score):>0.6f} | f0.5: {(100 * f05score):>0.6f}")
            snapshot_path = self.get_detailed_session_path(train_dataset, "weights", f"f{fold}", f"model_f{fold}.pth")
            self.save(snapshot_path)

    def test(self, test_dataset, weights_checkpoint_path):
        for path in weights_checkpoint_path:
            logger.info(f"testing checkpoint at: {path}")
            self.load_params(path)
            test_dataloader = DataLoader(test_dataset, batch_size=128)
            all_preds = []
            all_targets = []
            
            for X, y in test_dataloader:
                X = X.to_dense().cpu()
                y = y.cpu().squeeze()
                y_hat = self.svm.predict(X)
                y_hat = torch.tensor(y_hat)
                all_preds.extend(y_hat)
                all_targets.extend(y)
            all_preds = torch.stack(all_preds)
            all_targets = torch.stack(all_targets)
            base_path = "/".join(re.split("\\\|/", path)[:-1])
            with force_open(base_path + '/preds.pkl', 'wb') as file:
                pickle.dump(all_preds, file)
                logger.info(f'predictions are saved at: {file.name}')
            with force_open(base_path + '/targets.pkl', 'wb') as file:
                pickle.dump(all_targets, file)
                logger.info(f'targets are saved at: {file.name}')

    def get_dataloaders(self, dataset, train_ids, validation_ids, batch_size):
        train_subsampler = SubsetRandomSampler(train_ids)
        validation_subsampler = SubsetRandomSampler(validation_ids)
        train_loader = DataLoader(dataset, batch_size=len(train_ids),
            drop_last=False, sampler=train_subsampler)
        validation_loader = DataLoader(dataset, batch_size=len(validation_ids),
            drop_last=False, sampler=validation_subsampler)
        return train_loader, validation_loader

    def get_session_path(self, *args):
        return f"{self.session_path}" + self.__class__.short_name() + "/" + "/".join([str(a) for a in args])
    
    def get_new_optimizer(self, lr, *args, **kwargs):
        raise RuntimeError("you do not need to get an optimizer for this model")

    def get_detailed_session_path(self, dataset, *args):
        details = str(dataset) + "-" + str(self)
        return self.get_session_path(details, *args)
    
    def get_new_scheduler(self, optimizer, *args, **kwargs):
        raise RuntimeError("you do not need to get an scheduler for this model")
    
    def get_all_folds_checkpoints(self, dataset):
        main_path = glob(self.get_detailed_session_path(dataset, "weights", "f[0-9]", "model_f[0-9].pth")) # Supports upto 10 folds (from 0 to 9)
        paths = [ pp for pp in main_path if re.search(r"model_f\d{1,2}.pth$", pp)]
        if len(paths) == 0:
            raise RuntimeError("no checkpoint was found. probably the model has not been trained.")
        return paths
    
    def save(self, path):
        with force_open(path, "wb") as f:
            logger.info(f"saving sanpshot at {path}")
            pickle.dump(obj=self.svm, file=f)

    def load_params(self, path):
        with open(path, "rb") as f:
            logger.info(f"lading svm model from file: {path}")
            self.svm = pickle.load(f)
            logger.info("svm model loaded")
    
    def __str__(self) -> str:
        return 'svc-rbf'
    
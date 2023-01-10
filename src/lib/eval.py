import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score, precision_score, recall_score

def mse(target, pred):
    # https://pyimagesearch.com/2014/09/15/python-compare-two-images/
     target, pred = target.squeeze(), pred.squeeze()
     err = np.sum((target.astype("float") - pred.astype("float")) ** 2)
     err /= float(target.shape[0])# * target.shape[1])
     return err
    #return mean_squared_error(target, pred, squared=False)

def rmse(target, pred):
    return np.sqrt(mse(target, pred))
    #return mean_squared_error(target, pred, squared=True)

def evaluate(metrics, tgt, pred):
    metrics = [m for m in metrics if m != 'roc']
    r = dict().fromkeys(metrics)
    for m in r.keys(): r[m] = []
    if 'mse' in metrics: r['mse'].append(mse(tgt, pred))
    if 'rmse' in metrics: r['rmse'].append(rmse(tgt, pred))

    if 'precision_weighted' in metrics: r['precision_weighted'].append(precision_score(tgt, [p > 0.5 for p in pred], average='weighted'))
    if 'precision_micro' in metrics: r['precision_micro'].append(precision_score(tgt, [p > 0.5 for p in pred], average='micro'))
    if 'recall_weighted' in metrics: r['recall_weighted'].append(recall_score(tgt, [p > 0.5 for p in pred], average='weighted'))
    if 'recall_micro' in metrics: r['recall_micro'].append(recall_score(tgt, [p > 0.5 for p in pred], average='micro'))
    if 'F1_weighted' in metrics: r['F1_weighted'].append(f1_score(tgt, [p > 0.5 for p in pred], average='weighted'))
    if 'F1_micro' in metrics: r['F1_micro'].append(f1_score(tgt, [p > 0.5 for p in pred], average='micro'))

    return r
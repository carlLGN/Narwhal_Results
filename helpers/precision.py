import numpy as np
from sklearn.metrics import precision_score

def compute_precision(gt, pred):
    gt = gt.ravel().astype(bool)
    pred = pred.ravel().astype(bool)
    # Calculate precision score
    precision = precision_score(gt, pred)
    return precision
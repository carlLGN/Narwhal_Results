import numpy as np
from scipy.spatial.distance import dice
import medpy

def compute_dice(gt, pred):
    # Ensure the masks are boolean or binary
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    
    gt = np.reshape(gt, -1)
    pred = np.reshape(pred, -1)
    
    d=medpy.metric.dc(pred, gt)
    return d


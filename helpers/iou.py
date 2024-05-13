import numpy as np
from sklearn.metrics import jaccard_score


def compute_iou(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    
    mask1 = np.reshape(mask1, -1)
    mask2 = np.reshape(mask2, -1)
    
    j = jaccard_score(mask1, mask2)
    return j
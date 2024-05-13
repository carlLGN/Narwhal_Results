import numpy as np
from medpy.metric.binary import hd95

def compute_hd95(gt, pred, voxel_spacing = 0.325):
    # Ensure masks are boolean
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    
    # Calculate the Hausdorff 95 distance
    distance = hd95(pred, gt, voxel_spacing)
    return distance


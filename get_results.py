"""
Should be the final script for calculating quantitative results

Quantitative results to measure should include:

- Intersection over Union
- Dice Coefficient
- Hausdorff distance 95 (medpy)
"""

from helpers.iou import compute_iou
from helpers.dice import compute_dice
from helpers.hd95 import compute_hd95
from helpers.precision import compute_precision

from PIL import Image
import numpy as np
import os


labels_path = './data/labels/mask/'



# ANDERS MASK
#test_path = './data/AndersMask/mask/'



# TOSNETS
#test_path = './data/TOSNet/mask/'
#test_path = './data/TOSNet2/mask/'
#test_path = './data/TOSNet3/mask/'
#test_path = './data/TOSNetThin/mask/'
#test_path = './data/TOSNetThinLabelsEpoch9/mask/'
#test_path = './data/TOSNetThinLabelsEpoch19/mask/'
#test_path = './data/TOSNetThinLabelsEpoch29/mask/'
#test_path = './data/TOSNetLabelsEpoch9/mask/'
#test_path = './data/TOSNetLabelsEpoch19/mask/'
#test_path = './data/TOSNetLabelsEpoch29/mask/'
#test_path = './data/TOSNetLabelsEpoch39/mask/'
#test_path = './data/TOSNetZeroShotLabels/mask/'

#test_path = './data/TOSNetThinUNETEpoch19/mask/'
#test_path = './data/TOSNet128Epoch39/mask/'
#test_path = './data/TOSNet128Epoch19/mask/'
#test_path = './data/TOSNet128UNetEpoch39/mask/'
#test_path = './data/TOSNet128Epoch49/mask/'
#test_path = './data/TOSNet128Epoch29/mask/'

#test_path = './data/TOSNet128ThinEpoch49/mask/'
#test_path = './data/TOSNet128ThinEpoch29/mask/'
#test_path = './data/TOSNet128ThinEpoch19/mask/'


# UNETS
#test_path = './data/UNet/mask/'
#test_path = './data/UNetThinEpoch15/mask/'
#test_path = './data/UNetThinEpoch20/mask/'
#test_path = './data/UNetThinEpoch25/mask/'
#test_path = './data/UNetThinEpoch35/mask/'
#test_path = './data/UNetThickEpoch5/mask/'
#test_path = './data/UNetThickEpoch10/mask/'
#test_path = './data/UNetThickEpoch15/mask/'
#test_path = './data/UNetThickEpoch20/mask/'
#test_path = './data/UNetThickEpoch25/mask/'
#test_path = './data/UNetThickEpoch30/mask/'
#test_path = './data/UNetThickEpoch35/mask/'
test_path = './data/UNetThickEpoch40/mask/'





print(test_path + '\n')

if test_path[:13] == './data/TOSNet':

    labels_files = [f'mask_{i}.png' for i in range(18)]
    test_files = [f'Layer{i}.png' for i in range(18)]

    labels_list = [np.array(Image.open(os.path.join(labels_path, file))) for file in labels_files]
    labels_array = np.stack(labels_list, axis=0)

    test_list = [np.array(Image.open(os.path.join(test_path, file))) for file in test_files]
    test_list = [i[:,:,0] for i in test_list]
    test_array = np.stack(test_list, axis=0)

elif test_path[:11] == './data/UNet':

    labels_files = [f'mask_{i}.png' for i in range(18)]
    test_files = [f'mask_{i}.png' for i in range(18)]

    labels_list = [np.array(Image.open(os.path.join(labels_path, file))) for file in labels_files]
    labels_array = np.stack(labels_list, axis=0)

    test_list = [np.array(Image.open(os.path.join(test_path, file))) for file in test_files]
    test_array = np.stack(test_list, axis=0)

else:
    labels_files = [f'mask_{i}.png' for i in range(18)]
    test_files = [f'slice_{i}.png' for i in range(18)]

    labels_list = [np.array(Image.open(os.path.join(labels_path, file))) for file in labels_files]
    labels_array = np.stack(labels_list, axis=0)

    test_list = [np.array(Image.open(os.path.join(test_path, file))) for file in test_files]
    test_array = np.stack(test_list, axis=0)



print("IOU: " + str(compute_iou(labels_array, test_array)) + '\n')
print("Dice: " + str(compute_dice(labels_array, test_array)) + '\n')
print("HD95: " + str(compute_hd95(labels_array, test_array)) + '\n')
print("Precision: " + str(compute_precision(labels_array, test_array)))




'''
for i in range(18):
    

    labels_img = Image.open(labels_path + f'mask_{i}.png')
    test_img = Image.open(test_path + f'slice_{i}.png')
    


    labels_array = np.array(labels_img)
    test_array = np.array(test_img)

    print(f"Image: {i}")
    print("IOU: " + str(compute_iou(labels_array, test_array)) + '\n')
    print("Dice: " + str(compute_dice(labels_array, test_array)) + '\n')
    print("HD95: " + str(compute_hd95(labels_array, test_array)) + '\n\n')
'''

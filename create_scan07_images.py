import tifffile as tif
import numpy as np
from PIL import Image


def load_data(path):
    img = np.rollaxis(tif.imread(path), axis=1)
    return img

num = [140, 160, 203, 394, 323]
scan07_path = './data/scan_07_crop.tif'
path = './data/labels/img/'
data = load_data(scan07_path)

for i in num:
    arr = data[i]
    image = Image.fromarray(arr, 'L')
    image.save(path + f'slice_{i}.png')
    
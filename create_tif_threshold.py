import os
from PIL import Image
import numpy as np
import tifffile as tiff

def pngs_to_tiff(directory, output_path):
    # List all PNG files in the directory
    files = [f'Layer{i}.png' for i in range(18)]
    
    # Read the first image to find out the dimensions
    image_shape = Image.open(os.path.join(directory, files[0])).size
    # Create a numpy array to hold data from all images
    stack = np.zeros((len(files), image_shape[1], image_shape[0]), dtype=np.uint8)

    # Load each image as a numpy array and add it to the stack
    for i, filename in enumerate(files):
        img_path = os.path.join(directory, filename)
        with Image.open(img_path) as img:
            stack[i, :, :] = np.array(img)[:,:,0]
    
    # Save the stack as a 3D TIFF
    tiff.imwrite(output_path, stack, photometric='minisblack')


directory = './data/TOSNetThinLabelsEpoch19/mask/'  # Replace with the path to your folder of PNG images
output_path = 'tosnet_thin.tiff'  # Desired output file path
pngs_to_tiff(directory, output_path)

from PIL import Image
import cv2
import numpy as np

def remove_connected_components(mask, min_size=100):

    _, cc, stats, _ = cv2.connectedComponentsWithStats(mask)
    to_keep = np.squeeze(np.argwhere(stats[:,4]>min_size))
    for k in range(1,len(to_keep)):
        keep_val = to_keep[k]
        cc[cc==keep_val] = 255
    
    cc[cc<255] = 0
        
    return cc
    


num_images = 18
for i in range(num_images):
    image = Image.open(f'./data/labels/mask/mask_{i}.png')
    
    
    t = np.array(image)
    
    image.convert('L')
    mask = np.array(image)
    
    
    test = Image.fromarray(mask, 'L')
    test.save('bamtest.png')
    
    new_mask = remove_connected_components(mask, min_size=100).astype(np.uint8)

    image2 = Image.fromarray(new_mask)
    image2.save(f'./data/labels/test/mask_{i}.png')
    

"""
Instance of a UNet model with specific weights
"""

import monai
from monai.networks.nets import DynUNet
from monai.transforms import (
    Compose, LoadImaged, AddChanneld, Resized, ScaleIntensityd, ToTensord, RandRotate90d, AsDiscreted
)
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference

import torch
from PIL import Image
import numpy as np




class UNetModelInstance:
    
    def __init__(self, model, data, images, batch_size=1):
        self.images = images
        self.model = model
        self.data = data
        self.batch_size = batch_size
        self._transforms = infer_transforms = Compose([
        LoadImaged(keys=["img"]),
        AddChanneld(keys=["img"]),
        ScaleIntensityd(keys=["img"]),
        ToTensord(keys=["img"]),
        ])
        
        self.dataset = Dataset(self.data, transform=infer_transforms)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def get_mask(self, roi_size = (128,128), sw_batch_size=1, overlap=0.25):
        with torch.no_grad():
            for i, batch_data in enumerate(self.dataloader):
                inputs= batch_data['img'].to(self.device)
                outputs = sliding_window_inference(
                inputs=inputs,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=self.model,
                overlap=overlap
                )

                out = torch.argmax(outputs, dim=1)[0]

                array = (out.cpu().numpy() * 255).astype(np.uint8)
                array=array.T

                
                image = Image.fromarray(array, 'L')
                image.save(f'./data/UNetThinEpoch20/mask/mask_{images[i]}.png')


if __name__ == "__main__":
    
    
    labels_path = './data/labels/img/'
    images = [85]
    
    data = [
    {'img': labels_path + f'slice_{i}.png'} for i in images
    ]   
    
    unet_model = torch.load("./weights/UNet/Thin/UNetNarwhal_05-09-13-08_Epoch20.pth")
    instance = UNetModelInstance(unet_model, data, images)
    instance.get_mask(roi_size = (128,128), sw_batch_size=1, overlap=0.25)



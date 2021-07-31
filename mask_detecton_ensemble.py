'''
This mask detection is performed doing ensemble of 3 models developed by the team. 
We perform ensemble of models as:
  - Get prediction from all 3 models.
  - Chose the most confident pixels from each of these to form merged prediction.
  - From merged prediction, drop anything not atleast 20% confident.

Architecture and Implementation Detais of these models are available in project documentation.
'''

import matplotlib.pyplot as plt
from DL_FinalProject_Draft.inference import preTrainedResUnet_inference
from PIL import Image
import torchvision.transforms as transforms
import torch
from aerialDetection.inference import unet_inference
import os
import sys
import fastai
from fastai.vision.all import *
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from building_footprints_cs7643.losses.combined_loss import *
from building_footprints_cs7643.inference.inference import *


def model1_inference(img_tensor):
    inference = preTrainedResUnet_inference(output_channels=2)
    out = inference.run_inference(img_tensor) 
    building_pixels = torch.ones(out[0].shape)
    if inference.output_channels == 2:
        building_pixels = (torch.lt(out[0, :, :],out[1, :, :])).float()
    preds_model1 = building_pixels*out[1, :, :]    
    return preds_model1

def model2_inference(img_tensor):
    state_dict_path = './aerialDetection/model_state/project.pth'
    inference = unet_inference(state_dict_path)
    out = inference.run_inference(img_tensor)
    building_pixels = torch.ones(out[0].shape)
    if out.shape[0] == 2:
        building_pixels = (torch.lt(out[0, :, :],out[1, :, :])).float()
    preds_model2 = building_pixels*out[1, :, :]
    return preds_model2

def model3_inference(img_tensor):
    to_image = transforms.ToPILImage()
    img_PIL = to_image(img_tensor)
    out=run_inference3(img_PIL)
    building_pixels = torch.ones(out[0].shape)
    if out[2].shape[0] == 2:
        building_pixels = np.uint8(out[2][0]<out[2][1])
    preds_model3 = torch.tensor(building_pixels*out[2][1].numpy())    
    return preds_model3


def mask_inference(img_np, save_mask = True):
    ''' This does the mask inference by ensemling 3 models and merging them as described above '''

    img_np = img_np[:,:,:3]
    if not img_np.dtype == np.uint8:
        img_PIL = Image.fromarray(np.uint8(img_np*255))
    else:
        img_PIL = Image.fromarray(img_np)
    
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img_PIL)


    preds_model1 = model1_inference(img_tensor)
    preds_model2 = model2_inference(img_tensor)
    preds_model3 = model3_inference(img_tensor)

    pred_merged = torch.stack([preds_model1,preds_model2,preds_model3],dim=0)
    most_confident_pixels = pred_merged.max(dim=0)
    merged_mask = np.uint8(most_confident_pixels[0]>0.2)    

    if save_mask:
        plt.imsave('merged_mask.png',merged_mask)
    
    return merged_mask



if __name__ == "__main__":
    img_np = plt.imread('./demo.png')[:,:,:3]
    if not img_np.dtype == np.uint8:
        img_PIL = Image.fromarray(np.uint8(img_np*255))
    else:
        img_PIL = Image.fromarray(img_np)
    
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img_PIL)


    preds_model1 = model1_inference(img_tensor)
    preds_model2 = model2_inference(img_tensor)
    preds_model3 = model3_inference(img_tensor)

    pred_merged = torch.stack([preds_model1,preds_model2,preds_model3],dim=0)
    most_confident_pixels = pred_merged.max(dim=0)
    merged_mask = np.uint8(most_confident_pixels[0]>0.2)    

    plt.imsave('merged_mask.png',merged_mask)    
    print('mask saved as : merged_mask.png')
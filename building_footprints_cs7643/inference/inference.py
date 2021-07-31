import sys
sys.path.append('/home/sandeep/Desktop/CS7643_DeepLearning/Project/')
import fastai
from fastai.vision.all import *
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from building_footprints_cs7643.losses.combined_loss import *

path = Path('/home/sandeep/Desktop/CS7643_DeepLearning/Project/building_footprints_cs7643/')

def label_func(fn):
    ''' returns the path object for label file for passed image file path object '''
    return path/"imgs/masks-512-coded"/f"{fn.stem}{fn.suffix}"
    
def run_inference3(img_PIL):
    ''' runs the inference using the model loaded above '''

    codes = np.array(['background','building'])

    fnames = get_image_files(path/'imgs/images-512')

    aug_tfms = aug_transforms( do_flip=True, flip_vert=True, max_zoom=2,max_lighting=0.3,max_warp=0.1)

    dls = SegmentationDataLoaders.from_label_func(
        path, bs=1, fnames = fnames, 
        label_func = label_func, codes = codes, 
        item_tfms=[Resize(224)], batch_tfms=aug_tfms
        )

    learn = unet_learner(dls, resnet34, normalize=True, loss_func=CombinedLoss(), metrics=DiceMulti())

    learn.load('model_fastai_focal_dice_segmentation_2_40ep')

    img=np.asarray(img_PIL)
    out = learn.predict(img)    

    return out




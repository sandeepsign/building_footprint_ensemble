import fastai
from fastai.vision.all import *
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms


demo_im = Image.open('./demo.png')
demo_im_out = './demo_out.png'
to_tensor = transforms.ToTensor()
to_image = transforms.ToPILImage()

#strip transparency layer
# demo_im = demo_im[0:3, :, :]

state_dict_path = './models/model_fastai_focal_dice_segmentation_1_20ep.pth'


codes = np.array(['background','building'])

path = Path('/home/sandeep/Desktop/CS7643_DeepLearning/Project/building_footprints_cs7643/imgs/')
fnames = get_image_files(path/'images-512')

def label_func(fn):
    ''' returns the path object for label file for passed image file path object '''
    return path/"masks-512-coded"/f"{fn.stem}{fn.suffix}"

aug_tfms = aug_transforms( do_flip=True, flip_vert=True, max_zoom=2,max_lighting=0.3,max_warp=0.1)

dls = SegmentationDataLoaders.from_label_func(
       path, bs=1, fnames = fnames, 
       label_func = label_func, codes = codes, 
       item_tfms=[Resize(224)], batch_tfms=aug_tfms
    )

from fastai.vision.all import *

__all__ = ['DiceLoss', 'CombinedLoss']

def _one_hot(x, classes, axis=1):
    "Target mask to one hot"
    return torch.stack([torch.where(x==c, 1,0) for c in range(classes)], axis=axis)

class DiceLoss:
    "Dice coefficient metric for binary target in segmentation"
    def __init__(self, axis=1, smooth=1): 
        store_attr()
    def __call__(self, pred, targ):
        targ = _one_hot(targ, pred.shape[1])
        pred, targ = flatten_check(self.activation(pred), targ)
        inter = (pred*targ).sum()
        union = (pred+targ).sum()
        return 1 - (2. * inter + self.smooth)/(union + self.smooth)
    
    def activation(self, x): return F.softmax(x, dim=self.axis)
    def decodes(self, x):    return x.argmax(dim=self.axis)
    

class CombinedLoss:
    "Dice and Focal combined"
    def __init__(self, axis=1, smooth=1, alpha=1):
        store_attr()
        self.focal_loss = FocalLossFlat(axis=axis)
        self.dice_loss =  DiceLoss(axis, smooth)
        
    def __call__(self, pred, targ):
        return self.focal_loss(pred, targ) + self.alpha * self.dice_loss(pred, targ)
    
    def decodes(self, x):    return x.argmax(dim=self.axis)
    def activation(self, x): return F.softmax(x, dim=self.axis)

learn = unet_learner(dls, resnet34, normalize=True, loss_func=CombinedLoss(), metrics=DiceMulti())
learn.load('/home/sandeep/GoogleDrive/beans-home/mount_shared_partition/image_data/data_building_footprint/training_512x512/models/model_fastai_focal_dice_segmentation_1')

# inference = unet_inference(state_dict_path)
res = learn.predict(np.asarray(demo_im)[:,:,:3])
plt.imshow(res[2][1])

plt.imsave('demo_out.png',res[0])
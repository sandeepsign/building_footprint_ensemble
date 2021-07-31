#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 10:33:33 2021

@author: ahmedbilal
"""

import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import random
import os
from PIL import Image
import segmentation_models_pytorch as smp
from models import UNET
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms.functional as TF
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import copy
writer = SummaryWriter()
from ipywidgets import IntProgress


MODEL_PATH = '/home/sandeep/Desktop/CS7643_DeepLearning/Project/DL_FinalProject_Draft/best_model.pth'

#Switch to False if want to run the model from the start
load_model = True

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    


ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid'
# https://segmentation-modelspytorch.readthedocs.io/en/latest/


unet_build = UNET()
unet_select = unet_build.downs[:3]
#try other resnset backbones too 
model = smp.Unet('resnet34', 
                  classes=1, 
                  activation=ACTIVATION,
                  encoder_weights=ENCODER_WEIGHTS,
                  #https://arxiv.org/abs/1808.08127
                  decoder_attention_type = 'scse', #attention module used in decoder of the model
                  encoder_depth =5,
                  # decoder_channels = [256,128,64,32,16],
                  decoder_use_batchnorm=True,)
model_ = copy.deepcopy(model)


# Merge with UNET Build
model_.encoder.conv1 = unet_build.downs[:3]
model_.encoder.bn1 = Identity()
model_.encoder.relu = Identity()
model_.encoder.maxpool = Identity()
# print(summary(model,(3,224,224)))


if load_model:
    best_model = torch.load(MODEL_PATH)
    best_state_dict = best_model.state_dict()
    model.load_state_dict(best_state_dict)


resnet_pretrained = model
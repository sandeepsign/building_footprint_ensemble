import numpy as np
import torch
import torchvision

from torch.utils.tensorboard import SummaryWriter

import random

import os

from PIL import Image

from models import ResUNet
from losses import flatMSE, diceLoss

import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms

import torchvision.transforms.functional as TF

import copy

writer = SummaryWriter()


# absolute path to root of dataset. should contain images/ and gt/ folders
data_path = 'C:/aerialimagelabeling/AerialImageDataset'

NUM_EPOCHS = 20
BATCH_SIZE = 25

total_batch_count = 0

train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []

DEBUG = False
NUM_BLOCKS = 4
RESUME = True
RESUME_ACC = True
PRIME_CACHE = False

class AerialImageDataSet(Dataset):
    def __init__(self, path):
        self.im_path = '/'.join([path, 'images/'])
        self.gt_path = '/'.join([path, 'gt/'])
        self.im_cache_path = '/'.join([path, 'images_cache/'])
        self.gt_cache_path = '/'.join([path, 'gt_cache/'])
        self.num_ims = len(os.listdir(self.im_path))
        self.samples_per_img = 350

    def __len__(self):
        return self.samples_per_img*self.num_ims
    
    def __getitem__(self, idx):
        #putting 5000x5000 px into memory for 224x224 patch is slow. write 
        #patches to file once and only once and read patches into memory as
        #needed
        #check if cached version exists
        proposed_im_path = self.im_cache_path + str(idx) + '.pt'
        proposed_gt_path = self.gt_cache_path + str(idx) + '.pt'
        im_exists = os.path.isfile(proposed_im_path)
        gt_exists = os.path.isfile(proposed_gt_path)
        if im_exists and gt_exists:
            x = torch.load(proposed_im_path)
            y = torch.load(proposed_gt_path)
            return x, y
        #cached crop does not exist for at least 1 image, create and save to disk
        img_idx = idx // self.samples_per_img #get which 5000x5000 image should be used
        remainder = idx - img_idx * self.num_ims #index within 5000x5000 tile

        #we want this index to point to 1 image, so use seed for random
        random.seed(remainder) 
        #change aspect ratio of patch slightly
        aspect_ratio = random.uniform(0.9, 1.1)
        #change height of patch
        height = random.randint(100, 500)
        width = int(height * aspect_ratio)

        #get top left pixel of image.
        top = random.randint(1, 4999-height)
        left = random.randint(1, 4999-width)
        
        #randomly flip images
        h_flip = random.choice((False, True))
        v_flip = random.choice((False, True))


        #save to file
        x_name = os.listdir(self.im_path)[img_idx]
        x = Image.open('/'.join([self.im_path, x_name]))
        y = Image.open('/'.join([self.gt_path, x_name]))

        #use functional transforms for repeatability and identical ops on mask
        #and image
        x = TF.resized_crop(x, top, left, height, width, (224, 224), Image.NEAREST)
        y = TF.resized_crop(y, top, left, height, width, (224, 224), Image.NEAREST)
        if h_flip:
            x = TF.hflip(x)
            y = TF.hflip(y)
        if v_flip:
            x = TF.vflip(x)
            y = TF.vflip(y)
        x = TF.to_tensor(x)
        y = TF.to_tensor(y)
        y = y.type(torch.int8)
        #normalize to imagenet settings
        x = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        torch.save(x, proposed_im_path)
        torch.save(y, proposed_gt_path)

        return x, y


def accuracy(output, target):
    #calculate accuracy of 1xHwW network output vs 1xHxW target
    with torch.no_grad():
        best_guess = torch.lt(output[:, 0, :, :],output[:, 1, :, :]).long()
        flat_out = torch.flatten(best_guess)
        flat_targ = torch.flatten(target)
        abs_diff = torch.abs(torch.sub(flat_out, flat_targ))
        total_pixels = abs_diff.shape[0]
        misses = torch.sum(abs_diff)
        acc = (total_pixels-misses)/(total_pixels)
        return acc.item()

def train(epoch, data_loader, model, optimizer, criterion):
    #training loop
    model.train()
    global total_batch_count #for tensorboard global index
    accuracyNumerator = 0
    accuracyDenominator = 0
    lossNumerator = 0
    lossDenominator = 0
    for idx, (data, target) in enumerate(data_loader):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        #forward pass
        out = model.forward(data)
        num_entries = out.shape[0]
        loss = criterion.forward(out, target.long().squeeze(1))
        #track overall epoch loss
        sum_loss = out.shape[0] * loss
        lossNumerator += sum_loss.item()
        lossDenominator += num_entries
        #backward pass
        model.zero_grad()
        loss.backward()
        #step down gradient
        optimizer.step()
        #accuracy for this batch
        batch_acc = accuracy(out, target)
        num_pos = out.shape[0] * batch_acc
        accuracyNumerator += num_pos
        accuracyDenominator += num_entries
        #write batch stats to tensorboard
        writer.add_scalar('Loss/batch', loss.item(), total_batch_count)
        writer.add_scalar('Acc/batch', batch_acc, total_batch_count)
        total_batch_count += 1
    #overall epoch metrics
    train_loss_history.append(lossNumerator/lossDenominator)
    train_acc_history.append(accuracyNumerator/accuracyDenominator)
    print('Training Epoch #' + str(epoch)+' - Loss: ' + str(train_loss_history[-1]) + '; Acc: ' + str(train_acc_history[-1]))
    writer.add_scalar('Loss/train', train_loss_history[-1], epoch)
    writer.add_scalar('Acc/train', train_acc_history[-1], epoch)

def writeImage(out, targ, epoch):
    #write batch of images to tensorboard
    with torch.no_grad():
        roundedOut = torch.lt(out[:, 0, :, :],out[:, 1, :, :]).long().unsqueeze(1)
        gridOut = torchvision.utils.make_grid(roundedOut)
        gridTarg = torchvision.utils.make_grid(targ)
        writer.add_image('val_targ', gridTarg, epoch)
        writer.add_image('val_output', gridOut, epoch)

def validate(epoch, val_loader, model, criterion):
    #validation step
    model.eval()
    accuracyNumerator = 0
    accuracyDenominator = 0
    lossNumerator = 0
    lossDenominator = 0
    for idx, (data, target) in enumerate(val_loader):
        with torch.no_grad():
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            out = model.forward(data)
            loss = criterion.forward(out, target.long().squeeze(1))
            lossNumerator += out.shape[0] * loss.item()
            lossDenominator += out.shape[0]
            acc = accuracy(out, target)
            accuracyNumerator += out.shape[0] * acc
            accuracyDenominator += out.shape[0]
            if idx == 0:
                #print(out.shape)
                writeImage(out, target, epoch)
                prob = out[:, 1, :, :]
                writer.add_pr_curve('validation', target.squeeze(), prob, epoch)
    val_loss_history.append(lossNumerator/lossDenominator)
    val_acc_history.append(accuracyNumerator/accuracyDenominator)
    print('Validation Epoch #' + str(epoch)+' - Loss: ' + str(val_loss_history[-1]) + '; Acc: ' + str(val_acc_history[-1]))
    writer.add_scalar('Loss/val', val_loss_history[-1], epoch)
    writer.add_scalar('Acc/val', val_acc_history[-1], epoch)
    return val_acc_history[-1]





def main():
    #create dataset
    train_and_val_dataset = AerialImageDataSet('/'.join([data_path, 'train']))
    train_size = int(0.8*(len(train_and_val_dataset))) #80/20 train/val
    val_size = len(train_and_val_dataset) - train_size
    #train/val split
    torch.manual_seed(0)

    train_dataset, val_dataset = torch.utils.data.random_split(train_and_val_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    if PRIME_CACHE:
        for idx, (x, y) in enumerate(train_loader):
            if idx % 1000 == 0:
                print('train iteration: ' + str(idx))
            pass
        print('train cache complete')
        for idx, (x, y) in enumerate(val_loader):
            if idx % 1000 == 0:
                print('val iteration: ' + str(idx))
            pass
        print('val cache complete')
    model = ResUNet(num_blocks=NUM_BLOCKS)
    if torch.cuda.is_available():
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    bestAcc = 0
    best_model = None
    if RESUME: #loads last model state dict into memory
        print('loading previous best model and accuracy from file')
        best_state_dict = torch.load('./checkpoints/project.pth')
        model.load_state_dict(best_state_dict)
        best_model = copy.deepcopy(model)
        bestAcc = torch.load('./checkpoints/bestAcc.pth')
        print('accuracy to beat: ' + str(bestAcc*100) + '%')
    elif RESUME_ACC:
        print('loading previous best accuracy from file')
        bestAcc = torch.load('./checkpoints/bestAcc.pth')
        print('accuracy to beat: ' + str(bestAcc*100) + '%')
    for epoch in range(NUM_EPOCHS):
        train(epoch, train_loader, model, optimizer, criterion)

        accuracy = validate(epoch, val_loader, model, criterion)
        if accuracy > bestAcc:
            print('new best model with acc: ' + str(accuracy))
            bestAcc = accuracy
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), './checkpoints/project.pth')
            torch.save(bestAcc, './checkpoints/bestAcc.pth')
        print('Epoch num: ' + str(epoch)+'; Accuracy: ' + str(accuracy))
    writer.close()




if __name__ == '__main__':
    main()
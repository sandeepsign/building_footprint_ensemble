import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import random
import os
from PIL import Image
import segmentation_models_pytorch as smp
from models import UNET
from models.resunet34_pretrained import resnet_pretrained
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms.functional as TF
os.environ['KMP_DUPLICATE_LIB_OK']='True'
writer = SummaryWriter()


data_path = '/Users/ahmedbilal/Desktop/DL_Final_Project/AerialImageDataset'
NUM_EPOCHS = 20
BATCH_SIZE = 32
total_batch_count = 0
train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []
LEARNING_RATE = 1e-4


    
class AerialImageDataSet(Dataset):
    def __init__(self, path, samples = 65, preprocessing=None):
        self.im_path = '/'.join([path, 'images/'])
        self.gt_path = '/'.join([path, 'gt/'])
        self.im_cache_path = '/'.join([path, 'images_cache/'])
        self.gt_cache_path = '/'.join([path, 'gt_cache/'])
        self.num_ims = len(os.listdir(self.im_path))
        self.samples_per_img = samples
        self.preprocessing = preprocessing


    def __len__(self):
        return self.samples_per_img*self.num_ims
    
    def __getitem__(self, idx):
        #check if cached version exists
        proposed_im_path = self.im_cache_path + str(idx) + '.pt'
        proposed_gt_path = self.gt_cache_path + str(idx) + '.pt'
        im_exists = os.path.isfile(proposed_im_path)
        gt_exists = os.path.isfile(proposed_gt_path)
        if im_exists and gt_exists:
            x = torch.load(proposed_im_path)
            y = torch.load(proposed_gt_path)
            return x, y
        
        img_idx = idx // self.samples_per_img #get which 5000x5000 image should be used
        remainder = idx - img_idx * self.num_ims #index within 5000x5000 tile
        random.seed(remainder)
        aspect_ratio = random.uniform(0.9, 1.1)
        
        #get height,width, top and left
        height = random.randint(100, 500)
        width = int(height * aspect_ratio)
        top = random.randint(1, 4999-height)
        left = random.randint(1, 4999-width)
        #flip img or not, make more robust
        h_flip = random.choice((False, True))
        v_flip = random.choice((False, True))
        x_name = os.listdir(self.im_path)[img_idx]
        x = Image.open('/'.join([self.im_path, x_name]))
        y = Image.open('/'.join([self.gt_path, x_name]))
        #resized crop img
        x = TF.resized_crop(x, top, left, height, width, (224, 224), Image.NEAREST)
        y = TF.resized_crop(y, top, left, height, width, (224, 224), Image.NEAREST)
        
        #Augmentation
        if h_flip:
            x = TF.hflip(x)
            y = TF.hflip(y)
        if v_flip:
            x = TF.vflip(x)
            y = TF.vflip(y)
            
        x = TF.to_tensor(x)
        y = TF.to_tensor(y)
        y = y.type(torch.int8)
        x = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        torch.save(x, proposed_im_path)
        torch.save(y, proposed_gt_path)
        return x, y
    
    
def writeImage(out, targ, epoch):
    with torch.no_grad():
        roundedOut = torch.round(out)
        gridOut = torchvision.utils.make_grid(roundedOut)
        gridTarg = torchvision.utils.make_grid(targ)
        writer.add_image('val_targ', gridTarg, epoch)
        writer.add_image('val_output', gridOut, epoch)
        
             

def main():
    #create new train and val data set by sampling each img 350 times
    train_and_val_dataset = AerialImageDataSet('/'.join([data_path, 'train']))
    print('train dataset created')
    #train validation split
    train_size = int(0.8*(len(train_and_val_dataset)))
    val_size = len(train_and_val_dataset) - train_size
    torch.manual_seed(torch.initial_seed())
    train_dataset, val_dataset = torch.utils.data.random_split(train_and_val_dataset, [train_size, val_size])
    
    model = resnet_pretrained


    
    print('train dataset contains: ' + str(len(train_dataset)) + ' images')
    print('val dataset contains: ' + str(len(val_dataset)) + ' images')
    
    
    #make the dataloader object
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    print('train loader created')
    
    
    #Try different losses: https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/utils/losses.py
    loss = smp.utils.losses.DiceLoss()
    metrics = [
    smp.utils.metrics.Accuracy(),
    smp.utils.metrics.IoU(threshold=0.5),
    smp.utils.metrics.Precision(),
    smp.utils.metrics.Recall(),
    smp.utils.metrics.Fscore()
    ]
    

    optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
    ])
    
    # create epoch runners 
    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        verbose=True,
        )
    
    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        verbose=True,
        )


    max_score = 0
    train_ = []
    valid_ = []
    TRAIN = True
    if TRAIN==True:
        for i in range(NUM_EPOCHS):
            
            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader)
            for k,v in train_logs.items():
                writer.add_scalar(f'{k}/train', v, i)
            
            
            valid_logs = valid_epoch.run(val_loader)
            for k,v in valid_logs.items():
                writer.add_scalar(f'{k}/valid', v, i)
            for idx, (data, target) in enumerate(val_loader):
                if idx == 0:
                    out = model.forward(data)
                    writeImage(out, target, i)
                    writer.add_pr_curve('isBuildingPR', target, out, i, weights=None)
            
            if max_score < valid_logs['iou_score']:
                max_score = valid_logs['iou_score']
                torch.save(model, './best_model.pth')
                print('Model saved!')
            
            #Reduce LR on 8th epoch, feel free to change this
            if i == 8:
                optimizer.param_groups[0]['lr'] = 1e-5
                print('Decrease decoder learning rate to 1e-5')
            
            train_.append(train_logs)
            valid_.append(valid_logs)
    

if __name__ == '__main__':
    main()
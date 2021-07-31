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


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class diceLoss(nn.ModuleList):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        flat_x = torch.flatten(x)
        flat_y = torch.flatten(y)
        flat_y = flat_y.float()
        dot = torch.dot(flat_x, flat_y)
        num = torch.add(torch.mul(dot, 2),1)
        denom = torch.add(torch.add(flat_x.sum(), flat_y.sum()),1)
        loss = torch.sub(1, torch.div(num, denom))

        return loss
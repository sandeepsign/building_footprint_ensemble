import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class flatMSE(nn.ModuleList):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction='none')
        self.beta = 0.999999
    
    def forward(self, x, y):
        y = y.type(torch.float)
        flat_x = torch.flatten(x)
        flat_y = torch.flatten(y)
        total_y_1_px = torch.sum(flat_y)
        invert_y = torch.mul(flat_y, -1)
        total_y_0_px = torch.sum(torch.add(invert_y, 1))
        
        reverse_flat_y = torch.add(invert_y, 1)
        
        effective_1 = torch.div((1-torch.pow(self.beta, total_y_1_px)),1-self.beta)
        effective_0 = torch.div((1-torch.pow(self.beta, total_y_0_px)),1-self.beta)
        averageEffective = 0.5 * torch.add(effective_0,effective_1)

        weight = torch.mul(torch.add(torch.div(flat_y,effective_1),torch.div(reverse_flat_y, effective_0)), averageEffective)
        loss = self.loss(flat_x, flat_y)
        weighted_loss = torch.mul(weight, loss)
        weighted_loss = torch.mean(weighted_loss)
        return weighted_loss
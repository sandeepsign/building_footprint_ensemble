import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import models

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            #channels is the list of block output channels
            self, in_channels=3, out_channels=1, 
            channels=[16,32,64, 128, 256],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(0.25)
        #resnet34
        self.base_model = models.resnet34(pretrained=True)
        self.base_layers = list(self.base_model.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.resnet34_model = [self.layer0,self.layer1,self.layer2,self.layer3,self.layer4]
    
        # Down part of UNET
        for channel in channels:
            self.downs.append(DoubleConv(in_channels, channel))
            #update input channels with the output channels after each down layer
            in_channels = channel

        # Up part of UNET - starting from the bottom this time so need reversed 
        for channel in reversed(channels):
            self.ups.append(
                #This upsamples
                nn.ConvTranspose2d(
                    channel*2, channel, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(channel*2, channel))
            
        #this is the bridge layer, takes the last (512) channel layer from down and multiplies it by 2
        self.bridge = DoubleConv(channels[-1], channels[-1]*2)
        #final layer that maos
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels[0], out_channels, kernel_size=1),
            nn.Sigmoid()
                                        )

    def forward(self, x):
        skip_connections = []
  
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bridge(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            #doing convtranspose2d to upsample
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            #add the concat skip along the channels dimension
            x = torch.cat((skip_connection, x), dim=1)
            #halves the channels
            x = self.ups[idx+1](x)
        
            
            
        return self.final_conv(x)

def test():
    x = torch.randn((3, 3, 161, 161))
    model = UNET(in_channels=3, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)


if __name__ == "__main__":
    test()
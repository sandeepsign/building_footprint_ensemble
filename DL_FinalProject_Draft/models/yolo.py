"""
https://pjreddie.com/media/files/papers/YOLOv3.pdf
"""

import torch
import torch.nn as nn

""" 
Information about architecture config:
    
Tuple is structured by (filters, kernel_size, stride) 

Every conv is a same convolution. 

"B" indicating a residual block followed by the number of repeats

"S" is for scale prediction block and computing the yolo loss

"U" is for upsampling the feature map and concatenating with a previous layer
"""
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        #bn_act tells us whether the cnn block will be using a btch norm activation function
        super().__init__()
        #if using bn_act, we do not want to use a bias 
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        #for scale prediction (our output), we do not want to use leaky relu and batchnorm on our output
        #if using batch_norm activation
        if self.use_bn_act:
            #1) conv layer
            #2) batchnorm
            #3) leakyrelu
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    #use residual refers to 
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        #create a new layers list for this repeating block
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    #downsamples the number of filters and then brings it back
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]
        
        #keep track of whether this block requires skip connections - use residuals and number of repeats
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            #if we are using residuals for this residualblock, do the resnet thing of adding the input to the output at every step
            #if no residuals, just the output is fine
            #Due to reduce and enlarge, the output dimensions always equals input dimensions
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            #For every single cell, we have 3 anchor bounding boxes
            #For every bounding box, for each of our output classes, we want 5 values - [po (prob there is object in the cell), x,y,w,h ]
            CNNBlock(
                2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            #reshape to batch_size, no.of bounding boxes, set of values for each box, 
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            #now just rearrange the dimensions by bringing the 3rd dimension to the end
            .permute(0, 1, 3, 4, 2)
            #example output - N (batchsize), 3 (bouding boxes), 13 (bb height),13(bb width), 5 (5 output for each class)
        )


class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        #fill up the self.layers with conv and residual blocks
        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = []  # one output for each scale
        #route connections is a stack
        route_connections = [] #this is the pre scaled prediction output, save it for concat after scaled prediction
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                #add sigmoid here
                x = self.sigmoid(x)
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                #after upsample, concat the channels of upsampled x and last x output in route_connections
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        #we are just filling up the layers with cnn, residual blocks and predictions
        #module list is a list that pytorch tracks
        layers = nn.ModuleList()
        
        in_channels = self.in_channels

        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        #padding follows a simple pattern
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                #set out channels from this block to inchannels
                in_channels = out_channels
            
            #Residual Blocks - repeat conv layers stacked 
            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))

            elif isinstance(module, str):
                #If S compute the prediction for that scale
                if module == "S":
                    layers += [
                        #residual blocks maintains the same channels
                        #use residuals as false because we are not using skipped connections
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2
                
                #If U, need to upsample 
                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    #right after upsampling, we want to concatenate 3* the current image channels
                    in_channels = in_channels * 3

        return layers


# if __name__ == "__main__":
#     num_classes = 1
#     IMAGE_SIZE = 224
#     model = YOLOv3(num_classes=num_classes)
#     x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
#     out = model(x)
#     assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
#     assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
#     assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
#     print("Success!")
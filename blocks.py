import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(inp, oup, kernel_size, stride, upsampling = False, Leaky = False, InstanceNorm = False,
                conv_layer = nn.Conv2d, norm_layer = nn.BatchNorm2d, act_layer = nn.ReLU):
    if (upsampling):
        conv_layer = nn.ConvTranspose2d
    if (InstanceNorm):
        norm_layer = nn.InstanceNorm2d
    if (Leaky):
        act_layer = nn.LeakyReLU
        return nn.Sequential(
            #bias는 computational cost 줄이기위해 false
            conv_layer(inp, oup, kernel_size, stride, 1, bias = False),
            norm_layer(oup),
            act_layer(0.2, inplace = True)       
        )
    else:
        return nn.Sequential(
            #bias는 computational cost 줄이기위해 false
            conv_layer(inp, oup, kernel_size, stride, 1, bias = False),
            norm_layer(oup),
            act_layer(inplace = True)       
        )

def conv_3x3(inp, oup, stride = 1, conv_layer = nn.Conv2d):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias = False),
    )

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv_block1 = conv_bn(channels, channels, 3, 1)
        self.conv_block2 = nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, 1),
                nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = torch.add(x, residual)
        out = self.relu(x)
        return x
    
class SELayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class SEBasicBlock(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride=1, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv_block = conv_block(inp, oup, kernel_size, stride, Leaky = True, InstanceNorm = True)
        self.conv = conv_3x3(inp, oup)
        self.In = nn.InstanceNorm2d(oup)
        self.se = SELayer(oup, reduction)
        self.LReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = x
        out = self.conv_block(x)

        out = self.conv(out)
        out = self.In(out)
        out = self.se(out)
        '''
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        '''     
        out = self.LRELU(out)

        return out
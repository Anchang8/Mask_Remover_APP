import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import conv
from torch.nn.modules.batchnorm import BatchNorm2d

def conv_bn(inp, oup, kernel_size, stride, 
                conv_layer = nn.Conv2d, norm_layer = nn.BatchNorm2d, act_layer = nn.ReLU):
    return nn.Sequential(
        #bias는 computational cost 줄이기위해 false
        conv_layer(inp, oup, kernel_size, stride, 1, bias = False),
        norm_layer(oup),
        act_layer(inplace = True)       
    )

def conv_1x1_bn(inp, oup, conv_layer = nn.Conv2d, norm_layer = nn.BatchNorm2d, act_layer = nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias = False),
        norm_layer(oup),
        act_layer(inplace = True)
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
import torch
import torch.nn as nn
import torchvision.models as models
from mask.blocks import *

inp = 3
channels = 64

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.conv1 = SEBasicBlock(inp, channels, 4, 2)
        self.conv2 = SEBasicBlock(channels, channels * 2, 4, 2)
        self.conv3 = SEBasicBlock(channels * 2, channels * 4, 4, 2)
        self.conv4 = conv_block(channels * 4, channels * 6, 4, 2)
        self.conv5 = conv_block(channels * 6, channels * 8, 4, 2)
        
        self.deconv1 = conv_block(channels * 8, channels * 6, 4, 2, upsampling = True)
        self.deconv2 = conv_block(channels * 12, channels * 4, 4, 2, upsampling = True)
        self.deconv3 = conv_block(channels * 8, channels * 2, 4, 2, upsampling = True)
        self.deconv4 = conv_block(channels * 4, channels * 1, 4, 2, upsampling = True)
        self.deconv5 = nn.Sequential(
                nn.ConvTranspose2d(channels * 2, inp // 3, 4, 2, 1, bias = False),
                nn.Sigmoid()
        )
        
    def forward(self, x):
        out = self.conv1(x)
        out_256 = out
        
        out = self.conv2(out)
        out_128 = out
        
        out = self.conv3(out)
        out_64 = out
        
        out = self.conv4(out)
        out_32 = out
        
        out = self.conv5(out)
        
        out = self.deconv1(out)
        out = torch.cat([out, out_32], dim = 1)
        
        out = self.deconv2(out)
        out = torch.cat([out, out_64], dim = 1)
        
        out = self.deconv3(out)
        out = torch.cat([out, out_128], dim = 1)
        
        out = self.deconv4(out)
        out = torch.cat([out, out_256], dim = 1)
        
        out = self.deconv5(out)
        return out
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = SEBasicBlock(inp + 1, channels, 4, 2)
        self.conv2 = SEBasicBlock(channels, channels * 2, 4, 2)
        self.conv3 = SEBasicBlock(channels * 2, channels * 4, 4, 2)
        self.conv4 = conv_block(channels * 4, channels * 6, 4, 2)
        self.conv5 = conv_block(channels * 6, channels * 8, 4, 2)
        
        self.deconv1 = conv_block(channels * 8, channels * 6, 4, 2, upsampling = True)
        self.deconv2 = conv_block(channels * 12, channels * 4, 4, 2, upsampling = True)
        self.deconv3 = conv_block(channels * 8, channels * 2, 4, 2, upsampling = True)
        self.deconv4 = conv_block(channels * 4, channels * 1, 4, 2, upsampling = True)
        self.deconv5 = nn.Sequential(
                nn.ConvTranspose2d(channels * 2, inp, 4, 2, 1, bias = False),
                nn.Tanh()
        )
        
    def forward(self, x):
        out = self.conv1(x)
        out_256 = out
        
        out = self.conv2(out)
        out_128 = out
        
        out = self.conv3(out)
        out_64 = out
        
        out = self.conv4(out)
        out_32 = out
        
        out = self.conv5(out)
        
        out = self.deconv1(out)
        out = torch.cat([out, out_32], dim = 1)
        
        out = self.deconv2(out)
        out = torch.cat([out, out_64], dim = 1)
        
        out = self.deconv3(out)
        out = torch.cat([out, out_128], dim = 1)
        
        out = self.deconv4(out)
        out = torch.cat([out, out_256], dim = 1)
        
        out = self.deconv5(out)
        return out
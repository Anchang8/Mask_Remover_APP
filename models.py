import torch
import torch.nn as nn
import torchvision.models as models
from blocks import *

inp = 3
channels = 64

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.conv1 = conv_block(inp, channels, 4, 2)
        self.conv2 = conv_block(channels, channels * 2, 4, 2)
        self.conv3 = conv_block(channels * 2, channels * 4, 4, 2)
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
        out_512 = out
        
        out = self.conv2(out)
        out_256 = out
        
        out = self.conv3(out)
        out_128 = out
        
        out = self.conv4(out)
        out_64 = out
        
        out = self.conv5(out)
        
        out = self.deconv1(out)
        out = torch.cat([out, out_64], dim = 1)
        
        out = self.deconv2(out)
        out = torch.cat([out, out_128], dim = 1)
        
        out = self.deconv3(out)
        out = torch.cat([out, out_256], dim = 1)
        
        out = self.deconv4(out)
        out = torch.cat([out, out_512], dim = 1)
        
        out = self.deconv5(out)
        return out
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
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
                nn.ConvTranspose2d(channels * 2, inp, 4, 2, 1, bias = False),
                nn.Tanh()
        )
        
    def forward(self, x):
        out = self.conv1(x)
        out_128 = out
        
        out = self.conv2(out)
        out_64 = out
        
        out = self.conv3(out)
        out_32 = out
        
        out = self.conv4(out)
        out_16 = out
        
        out = self.conv5(out)
        
        out = self.deconv1(out)
        out = torch.cat([out, out_16], dim = 1)
        
        out = self.deconv2(out)
        out = torch.cat([out, out_32], dim = 1)
        
        out = self.deconv3(out)
        out = torch.cat([out, out_64], dim = 1)
        
        out = self.deconv4(out)
        out = torch.cat([out, out_128], dim = 1)
        
        out = self.deconv5(out)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(inp, channels, 4, 2, 1, bias = False),
                nn.LeakyReLU(0.2, inplace = True)
        )
        self.conv2 = conv_block(channels, channels * 2, 4, 2, Leaky = True, InstanceNorm = True)
        self.conv3 = conv_block(channels * 2, channels * 4, 3, 1, Leaky = True, InstanceNorm = True)
        self.conv4 = conv_block(channels * 4, channels * 8, 3, 1, Leaky = True, InstanceNorm = True)
        self.conv5 = nn.Sequential(
            nn.Conv2d(channels * 8, 1, 3, 1, 1, bias = False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        return out
    
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        for x in range(18):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 27):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(27, 36):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu3_4 = h
        h = self.slice2(h)
        h_relu4_4 = h
        h = self.slice3(h)
        h_relu5_4 = h
        out = [h_relu3_4, h_relu4_4, h_relu5_4]
        return out
import numpy as np
import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

def rand_fix(random_seed = 777):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    
def save_checkpoint(state, dirpath, epoch):
    filename = 'checkpoint-{}.pt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    torch.save(state, checkpoint_path)
    print('--- checkpoint saved to ' + str(checkpoint_path) + ' ---')

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.utils.spectral_norm(m)
        
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
def trans(mode = 'normal'):
    imsize = 512
    if mode == 'normal':
        transform = transforms.Compose([
            #transforms.ColorJitter(brightness = (0.5,2)),
            transforms.Resize((imsize,imsize)),
            transforms.ToTensor()
        ])
        return transform 

    if mode == 'mask':
        transform_mask = transforms.Compose([
            transforms.Resize((imsize,imsize)),
            transforms.ToTensor()
        ])
        return transform_mask
    
    if mode == 'test':
        transform_test = transforms.Compose([
            transforms.Resize((imsize,imsize)),
            transforms.ToTensor()
        ])
        return transform_test
import numpy as np
import time
import random
import os
import matplotlib.pyplot as plt
from IQA_pytorch import SSIM

import torch
import torch.optim as optim
import torchvision.models as vmodels
from torchvision import utils
from torch.utils.data import Dataset, DataLoader

from dataloader import *
from models import *
from utils import *

random_seed = 777

rand_fix(random_seed)

device = torch.device("cuda:4" if (torch.cuda.is_available()) else "cpu")

vgg = VGG19(requires_grad=False).to(device)

epoch = 16
dataset_dir = "./Dataset/"
save_dir = "./CheckPoint3/"
model_dir = './CheckPoint3/checkpoint-{}.pt'.format(epoch)
Epoch = 19
bin_model_dir = './binaryzation_checkpoint/checkpoint-{}.pt'.format(Epoch)

num_workers = 0
batch_size = 2
num_epochs = 50
lr_G = 0.001
lr_D = 0.003
real_label = 1.
fake_label = 0.
alpha = 1
beta = 1
gamma = 100

transform = trans(mode = 'normal')
#transform_mask = trans(mode = 'mask')

train_dataset = FaceMask(dataset_dir, transform)
train_dataloader = DataLoader(train_dataset, batch_size = batch_size,
                             shuffle = True, num_workers = num_workers)

test_dataset = FaceMask(dataset_dir, transform, test = True)
test_dataloader = DataLoader(test_dataset, batch_size = len(test_dataset),
                            shuffle = False, num_workers = num_workers)

sample = next(iter(test_dataloader))
test_img = sample['test_img'].to(device)
show_img = sample['show_img']
'''
checkpoint = torch.load(model_dir)
netG_state_dict = checkpoint['netG_state_dict']
netD_state_dict = checkpoint['netD_state_dict']
optG_state_dict = checkpoint['G_opt']
optD_state_dict = checkpoint['D_opt']
'''
netG = Generator().to(device)
netG.apply(weights_init)
#netG.load_state_dict(netG_state_dict)

netD = Discriminator().to(device)
netD.apply(weights_init)
#netD.load_state_dict(netD_state_dict)


checkpoint = torch.load(bin_model_dir)
bin_model_state_dict = checkpoint['model_state_dict']
bin_model = Unet().to(device)
bin_model.apply(weights_init)
bin_model.load_state_dict(bin_model_state_dict)
bin_model.eval()

test_bin_img = bin_model(test_img)
test_img_cat = torch.cat([test_img, test_bin_img], dim = 1).to(device)

optim_G = optim.Adam(netG.parameters(), lr = lr_G, betas = (0.5, 0.999))
#optim_G.load_state_dict(optG_state_dict)

optim_D = optim.Adam(netD.parameters(), lr = lr_D, betas = (0.5, 0.999))
#optim_D.load_state_dict(optD_state_dict)

dataloader = train_dataloader

criterion = nn.L1Loss()
gan_loss = nn.BCELoss()
ssim = SSIM()

G_losses = []
D_losses = []
shape_losses = []
perceptual_losses = []
SSIM_losses = []

Start = time.time()
print("Starting Training Loop...")

for epoch in range(num_epochs):
    netG.train()
    netD.train()
    print("Epoch {}/{}".format(epoch + 1, num_epochs))
    print("-" * 10)
    start = time.time()
    
    face_masked = None
    face_unmasked = None
    for i, sample in enumerate(dataloader, 0):
        face_masked, face_unmasked = sample['W_mask'], sample['WO_mask']
        batch_size = face_masked.size(0)
        
        face_masked = face_masked.to(device)
        face_unmasked = face_unmasked.to(device)
        
        optim_D.zero_grad()
        optim_G.zero_grad()
        
        #Update D with GAN_Loss
        binaryzation_mask = bin_model(face_masked)
        binaryzation_mask = binaryzation_mask.to(device)    
        face_masked_cat = torch.cat([face_masked, binaryzation_mask], dim = 1)
        
        output = netG(face_masked_cat)
        fake = netD(output)        
        real = netD(face_unmasked)

        Real_label = torch.full((real.size()), real_label, dtype = torch.float, device = device)
        Fake_label = torch.full((fake.size()), fake_label, dtype = torch.float, device = device)
        D_Loss = (gan_loss(real, Real_label) + gan_loss(fake, Fake_label)) * alpha
        # Loss 설정시 L(pre, target) 왼쪽은 prediction 오른쪽은 target
        D_losses.append(D_Loss.item())
        
        #Update G with GAN_Loss
        G_Loss = (gan_loss(fake, Real_label)) * alpha
        G_losses.append(G_Loss.item())
        G_Loss.backward(retain_graph=True)
        
        #Update G with Shape_Loss
        Shape_Loss = criterion(output, face_unmasked) * beta
        shape_losses.append(Shape_Loss.item())
        
        gt1_relu3_4, gt1_relu4_4, gt1_relu5_4 = vgg(face_unmasked)
        output_relu3_4, output_relu4_4, output_relu5_4 = vgg(output)
        
        perceptual_loss = criterion(gt1_relu3_4, output_relu3_4) + criterion(gt1_relu4_4, output_relu4_4) + criterion(gt1_relu5_4, output_relu5_4)
        perceptual_losses.append(perceptual_loss.item())
        
        SSIM_loss = ssim(output, face_unmasked)
        SSIM_losses.append(SSIM_loss.item())
        
        Loss = (D_Loss + G_Loss) * alpha + Shape_Loss * beta + (perceptual_loss + SSIM_loss) * gamma
        Loss.backward()
        optim_D.step()
        optim_G.step()
        
        if (i % 1000 == 0):
                print("[{:d}/{:d}] D_ganL:{:.4f}     G_ganL:{:.4f}     Shape_L:{:.4f}    Perceptual_L:{:.4f}    SSIM_L:{:.4f}".
             format(i, len(dataloader), D_Loss.item(), G_Loss.item(), Shape_Loss.item(), perceptual_loss.item(), SSIM_loss.item()))
    
    save_checkpoint({
            'epoch' : epoch + 1,
            'netG_state_dict' : netG.state_dict(),
            'netD_state_dict' : netD.state_dict(),
            'G_opt' : optim_G.state_dict(),
            'D_opt' : optim_D.state_dict()
    }, save_dir, epoch + 1)
    
    print("="*100)
    minutes = (time.time() - start) // 60
    print('Time taken by epoch: {:.0f}h {:.0f}m {:.0f}s'.format(minutes // 60, minutes, (time.time() - start) % 60))
    print()
    
    with torch.no_grad():
        netG.eval()
        result = netG(face_masked_cat).cpu()
        test_result = netG(test_img_cat).cpu()
        
        inp = face_masked.cpu()
        oup = face_unmasked.cpu()
        
        sample = []
        test_sample = []
        
        for i in range(batch_size):
            sample.extend([inp[i], result[i]])
        for i in range(test_img.size(0)):
            test_sample.extend([test_img[i].cpu(), test_result[i]])
        
        result_img = utils.make_grid(sample, padding = 2,
                                        normalize = True, nrow = 2)
        test_result_img = utils.make_grid(test_sample, padding = 0,
                                        normalize = True, nrow = 2)
        utils.save_image(result_img, "./result3/result-{}epoch.png".format(epoch + 1))
        utils.save_image(test_result_img, "./result3/test_result-{}epoch.png".format(epoch + 1))
        
print("Training is finished")
minutes = (time.time() - Start) // 60
print('Time taken by num_epochs: {:.0f}h {:.0f}m {:.0f}s'.format(minutes // 60, minutes, (time.time() - Start) % 60))
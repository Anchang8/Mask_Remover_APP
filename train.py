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

dataset_dir = "./Datasets/"
dataset_dir_made = './dataset'
save_dir = "./CheckPoint5/"
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
transform_mask = trans(mode = 'mask')
transform_test = trans(mode = 'test')

train_dataset = FaceMask(dataset_dir, dataset_dir_made, transform, transform_mask)
train_dataloader = DataLoader(train_dataset, batch_size = batch_size,
                             shuffle = True, num_workers = num_workers)

test_dataset = FaceMask(dataset_dir, dataset_dir_made, transform, transform_test = transform_test, test = True)
test_dataloader = DataLoader(test_dataset, batch_size = 8,
                            shuffle = False, num_workers = num_workers)

sample = next(iter(test_dataloader))
test_img = sample['test_img'].to(device)
show_img = sample['show_img']

netG = Generator().to(device)
netG.apply(weights_init)
netD = Discriminator().to(device)
netD.apply(weights_init)
ssim = SSIM()

optim_G = optim.Adam(netG.parameters(), lr = lr_G, betas = (0.5, 0.999))
optim_D = optim.Adam(netD.parameters(), lr = lr_D, betas = (0.5, 0.999))

dataloader = train_dataloader
netG.train()
netD.train()

criterion = nn.L1Loss()
gan_loss = nn.BCELoss()

G_losses = []
D_losses = []
shape_losses = []
perceptual_losses = []
SSIM_losses = []

Start = time.time()
print("Starting Training Loop...")

for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch + 1, num_epochs))
    print("-" * 10)
    start = time.time()
    
    face_masked = None
    face_unmasked = None
    
    face_cloth = None
    face_surgical = None
    face_unmasked2 = None
    for i, sample in enumerate(dataloader, 0):
        face_masked, face_unmasked = sample['W_mask'], sample['WO_mask']
        face_cloth, face_surgical, face_unmasked2 = sample['cloth'], sample['surgical'], sample['gt']
        batch_size = face_masked.size(0)
        
        face_masked = face_masked.to(device)
        face_unmasked = face_unmasked.to(device)
        face_cloth = face_cloth.to(device)
        face_surgical = face_surgical.to(device)
        face_unmasked2 = face_unmasked2.to(device)
        
        optim_D.zero_grad()
        optim_G.zero_grad()
        
        #Update D with GAN_Loss
        output = netG(face_masked)
        output_c = netG(face_cloth)
        output_s = netG(face_surgical)
        
        fake = netD(output.detach())
        fake_c = netD(face_cloth.detach())
        fake_s = netD(face_surgical.detach())
        
        real = netD(face_unmasked)
        real2 = netD(face_unmasked2)
        Real_label = torch.full((real.size()), real_label, dtype = torch.float, device = device)
        Fake_label = torch.full((fake.size()), fake_label, dtype = torch.float, device = device)
        D_Loss = (gan_loss(real, Real_label) + gan_loss(fake, Fake_label))
        D_Loss_made = (gan_loss(real2, Real_label) + gan_loss(fake_c, Fake_label) + gan_loss(fake_s, Fake_label))
        
        D_Loss = (D_Loss + D_Loss_made) * alpha
        # Loss 설정시 L(pre, target) 왼쪽은 prediction 오른쪽은 target
        D_losses.append(D_Loss.item())
        
        #Update G with GAN_Loss
        G_Loss = (gan_loss(fake, Real_label) + gan_loss(fake_c, Real_label) + gan_loss(fake_s, Real_label)) * alpha
        G_losses.append(G_Loss.item())
        G_Loss.backward(retain_graph=True)
        
        #Update G with Shape_Loss
        Shape_Loss = criterion(output, face_unmasked) + criterion(output_c, face_unmasked2) + criterion(output_s, face_unmasked2)
        Shape_Loss = Shape_Loss * beta
        shape_losses.append(Shape_Loss.item())
        
        gt1_relu3_4, gt1_relu4_4, gt1_relu5_4 = vgg(face_unmasked)
        gt2_relu3_4, gt2_relu4_4, gt2_relu5_4 = vgg(face_unmasked2)
        output_relu3_4, output_relu4_4, output_relu5_4 = vgg(output)
        outputC_relu3_4, outputC_relu4_4, outputC_relu5_4 = vgg(output_c)
        outputS_relu3_4, outputS_relu4_4, outputS_relu5_4 = vgg(output_s)
        
        perceptual_loss1 = criterion(gt1_relu3_4, output_relu3_4) + criterion(gt1_relu4_4, output_relu4_4) + criterion(gt1_relu5_4, output_relu5_4)
        perceptual_loss2 = criterion(gt2_relu3_4, outputC_relu3_4) + criterion(gt2_relu4_4, outputC_relu4_4) + criterion(gt2_relu5_4, outputC_relu5_4)
        perceptual_loss3 = criterion(gt2_relu3_4, outputS_relu3_4) + criterion(gt2_relu4_4, outputS_relu4_4) + criterion(gt2_relu5_4, outputS_relu5_4)
        
        perceptual_loss = perceptual_loss1 + perceptual_loss2 + perceptual_loss2
        perceptual_losses.append(perceptual_loss.item())
        
        SSIM1 = ssim(output, face_unmasked)
        SSIMc = ssim(output_c, face_unmasked2)
        SSIMs = ssim(output_s, face_unmasked2)
        
        SSIM_loss = (SSIM1 + SSIMc + SSIMs)
        SSIM_losses.append(SSIM_loss.item())
        
        Loss = (D_Loss + G_Loss) * alpha + Shape_Loss * beta + (perceptual_loss + SSIM_loss) * gamma
        Loss.backward()
        optim_D.step()
        optim_G.step()
        
        if (i % 200 == 0):
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
    print('Time taken by epoch: {:.0f}h {:.0f}m {:.0f}s'.format((
        (time.time() - start) // 60) // 60, (time.time() - start) // 60, (time.time() - start) % 60))
    print()
    
    with torch.no_grad():
        result = netG(face_masked).cpu()
        result_s = netG(face_surgical).cpu()
        result_c = netG(face_cloth).cpu()
        test_result = netG(test_img).cpu()
        
        inp = face_masked.cpu()
        oup = face_unmasked.cpu()
        oup2 = face_unmasked2.cpu()
        
        sample = []
        test_sample = []
        
        for i in range(batch_size):
            sample.extend([oup[i], result[i], result_s[i], result_c[i], oup2[i]])
        for i in range(8):
            test_sample.extend([test_img[i].cpu(), test_result[i]])
        
        result_img = utils.make_grid(sample, padding = 2,
                                        normalize = True, nrow = 5)
        test_result_img = utils.make_grid(test_sample, padding = 0,
                                        normalize = True, nrow = 2)
        utils.save_image(result_img, "./result5/result-{}epoch.png".format(epoch + 1))
        utils.save_image(test_result_img, "./result5/test_result-{}epoch.png".format(epoch + 1))
        
print("Training is finished")
hour = ((time.time() - Start) // 60) // 60
print('Time taken by num_epochs: {:.0f}h {:.0f}m {:.0f}s'.format(hour, (time.time() - Start) - hour * 60, (time.time() - Start) % 60))
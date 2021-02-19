import numpy as np
import time
import random

import torch
import torch.optim as optim
from torchvision import utils
from torch.utils.data import Dataset, DataLoader

from dataloader import *
from models import *
from utils import *

random_seed = 777

rand_fix(random_seed)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

dataset_dir = "../Mask_Removal_GAN1/Datasets/"
save_dir = "./CheckPoint/"
num_workers = 0
batch_size = 4
num_epochs = 20
lr_G = 0.001
lr_D = 0.003
real_label = 1.
fake_label = 0.
alpha = 1.
beta = 1.

transform = trans(mode = 'normal')
transform_mask = trans(mode = 'mask')
transform_test = trans(mode = 'test')

train_dataset = FaceMask(dataset_dir, transform, transform_mask)
train_dataloader = DataLoader(train_dataset, batch_size = batch_size,
                             shuffle = True, num_workers = num_workers)

test_dataset = FaceMask(dataset_dir, transform, transform_test = transform_test, test = True)
test_dataloader = DataLoader(test_dataset, batch_size = batch_size,
                            shuffle = False, num_workers = num_workers)

sample = next(iter(test_dataloader))
test_img = sample['test_img'].to(device)
show_img = sample['show_img']

netG = Generator().to(device)
netG.apply(weights_init)
netD = Discriminator().to(device)
netD.apply(weights_init)

optim_G = optim.Adam(netG.parameters(), lr = lr_G, betas = (0.5, 0.999))
optim_D = optim.Adam(netD.parameters(), lr = lr_D, betas = (0.5, 0.999))

dataloader = train_dataloader
netG.train()
netD.train()

shape_loss = nn.L1Loss()
gan_loss = nn.BCELoss()

G_losses = []
D_losses = []
shape_losses = []

Start = time.time()
print("Starting Training Loop...")

for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch + 1, num_epochs))
    print("-" * 10)
    start = time.time()
    
    face_masked = None
    mask = None
    for i, sample in enumerate(dataloader, 0):
        face_masked, face_unmasked = sample['W_mask'], sample['WO_mask']
        batch_size = face_masked.size(0)
        
        face_masked = face_masked.to(device)
        face_unmasked = face_unmasked.to(device)
        
        optim_D.zero_grad()
        optim_G.zero_grad()
        
        #Update D with GAN_Loss
        output = netG(face_masked)
        fake = netD(output.detach())
        real = netD(face_unmasked)
        Real_label = torch.full((real.size()), real_label, dtype = torch.float, device = device)
        Fake_label = torch.full((fake.size()), fake_label, dtype = torch.float, device = device)
        D_Loss = gan_loss(Real_label, real) + gan_loss(Fake_label, fake) * alpha
        D_losses.append(D_Loss.item())
        D_Loss.backward()
        optim_D.step()
        
        #Update G with GAN_Loss
        G_Loss = gan_loss(Real_label, fake) * alpha
        G_losses.append(G_Loss.item())
        G_Loss.backward()
        optim_G.step()
     
        optim_G.zero_grad()
        
        #Update G with Shape_Loss
        Shape_Loss = shape_loss(output, face_unmasked) * beta
        shape_losses.append(Shape_Loss.item())
        Shape_Loss.backward()
        optim_G.step()
        
        if (i % 1000 == 0):
                print("[{:d}/{:d}] D_ganL:{:.4f}     G_ganL:{:.4f}     Shape_L:{:.4f}".
             format(i, len(dataloader), D_Loss.item(), G_Loss.item(), Shape_Loss.item()))
        
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
        test_result = netG(test_img).cpu()
        inp = face_masked.cpu()
        oup = face_unmasked.cpu()
        sample = []
        test_sample = []
        
        for i in range(batch_size):
            sample.extend([inp[i], oup[i], result[i]])
            test_sample.append(test_img[i], test_result[i])
        
        result_img = utils.make_grid(sample, padding = 2,
                                        normalize = True, nrow = 3)
        test_result_img = utils.make_grid(test_sample, padding = 0,
                                        normalize = True, nrow = 2)
        utils.save_image(result_img, "./result/result-{}epoch.png".format(epoch + 1))
        utils.save_image(test_result_img, "./result/test_result-{}epoch.png".format(epoch + 1))
        
print("Training is finished")
hour = ((time.time() - Start) // 60) // 60
print('Time taken by num_epochs: {:.0f}h {:.0f}m {:.0f}s'.format(hour, (time.time() - Start) - hour * 60, (time.time() - Start) % 60))
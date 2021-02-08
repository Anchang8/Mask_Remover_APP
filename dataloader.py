from PIL import Image
import random
import os

import torch

class FaceMask():
    def __init__(self, dataset_dir, transform = None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        #Sorting by name
        self.Wmask_img = list(sorted(os.listdir(os.path.join(dataset_dir, 'train/with_mask/'))))
        self.WOmask_img = list(sorted(os.listdir(os.path.join(dataset_dir, 'train/without_mask/'))))
    
    def __len__(self):
        dir_path = os.path.join(self.dataset_dir, 'train/with_mask')
        length = len(os.walk(dir_path).__next__()[2])
        return length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        Wmask_img_path = os.path.join(self.dataset_dir, 'train/with_mask/', self.Wmask_img[idx])
        WOmask_img_path = os.path.join(self.dataset_dir, 'train/without_mask/', self.WOmask_img[idx])

        Wmask_img = Image.open(Wmask_img_path)
        WOmask_img = Image.open(WOmask_img_path)

        if self.transform:
            Wmask_img = self.transform(Wmask_img)
            WOmask_img = self.transform(WOmask_img)

        sample = {'WO_mask' : WOmask_img, 'W_mask' : Wmask_img}

        return sample
>>>>>>> 488bf70dd6f0f394847c812dea5107b3405316f1

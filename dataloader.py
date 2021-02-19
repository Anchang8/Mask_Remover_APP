from PIL import Image
import random
import os

import torch
from torchvision import transforms as trans

class FaceMask():
    def __init__(self, dataset_dir, transform = None, transform_mask = None, transform_test = None, test = False):
        self.test = test
        self.rand = float(random.randint(-15,15))
        self.transform_resize = trans.Resize((1024,1024))
        if self.test:
            self.dataset_dir = dataset_dir + 'test/'
            self.test_img = list(sorted(os.listdir(os.path.join(self.dataset_dir))))
        else:
            self.dataset_dir = dataset_dir + 'train/'
            self.Wmask_img = list(sorted(os.listdir(os.path.join(self.dataset_dir, 'with_mask/'))))
            self.WOmask_img = list(sorted(os.listdir(os.path.join(self.dataset_dir, 'without_mask/'))))
            self.mask_img = list(sorted(os.listdir(os.path.join(self.dataset_dir, 'mask/'))))
        
        self.transform_mask = transform_mask
        self.transform = transform
        self.transform_test = transform_test
    
    def __len__(self):
        if self.test:
            dir_path = os.path.join(self.dataset_dir)
        else:
            dir_path = os.path.join(self.dataset_dir, 'with_mask')
        length = len(os.walk(dir_path).__next__()[2])
        return length

    def __getitem__(self, idx):
        if self.test:
            test_img_path = os.path.join(self.dataset_dir, self.test_img[idx])
            test_img = Image.open(test_img_path).convert('RGB')
            show_img = self.transform_test(test_img)
            
            test_img = self.transform_resize(test_img)
            test_img = self.transform(test_img)
            
            sample = {'test_img' : test_img, 'show_img' : show_img}
            
        else:
            Wmask_img_path = os.path.join(self.dataset_dir, 'with_mask/', self.Wmask_img[idx])
            WOmask_img_path = os.path.join(self.dataset_dir, 'without_mask/', self.WOmask_img[idx])
            mask_img_path = os.path.join(self.dataset_dir, 'mask/', self.mask_img[idx])

            Wmask_img = Image.open(Wmask_img_path)
            WOmask_img = Image.open(WOmask_img_path)
            mask_img = Image.open(mask_img_path).convert("L")

            mask_img = trans.functional.rotate(mask_img, angle = self.rand, fill = (0,))
            mask_img = self.transform_mask(mask_img)
            mask_img = mask_img.float()

            Wmask_img = self.transform_resize(Wmask_img)
            Wmask_img = trans.functional.rotate(Wmask_img, angle = self.rand)
            Wmask_img = self.transform(Wmask_img)
                
            WOmask_img = self.transform_resize(WOmask_img)
            WOmask_img = trans.functional.rotate(WOmask_img, angle = self.rand)
            WOmask_img = self.transform(WOmask_img)
   
            sample = {'WO_mask' : WOmask_img, 'W_mask' : Wmask_img, 'mask' : mask_img}

        return sample
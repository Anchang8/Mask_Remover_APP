from PIL import Image
import random
import os

import torch
from torchvision import transforms as trans

class FaceMask():
    def __init__(self, dataset_dir, dataset_dir_made, transform = None, transform_mask = None, transform_test = None, test = False):
        self.test = test
        self.transform_resize = trans.Resize((256, 256))
        if self.test:
            self.dataset_dir = dataset_dir + 'test/'
            self.test_img = list(sorted(os.listdir(os.path.join(self.dataset_dir))))
        else:
            self.dataset_dir = dataset_dir + 'train/'
            self.dataset_dir_made = dataset_dir_made
            self.Wmask_img = list(sorted(os.listdir(os.path.join(self.dataset_dir, 'with_mask/'))))
            self.WOmask_img = list(sorted(os.listdir(os.path.join(self.dataset_dir, 'without_mask/'))))
            self.mask_img = list(sorted(os.listdir(os.path.join(self.dataset_dir, 'mask/'))))
            self.gt = list(sorted(os.listdir(os.path.join(self.dataset_dir_made, 'gt/'))))
            self.cloth = list(sorted(os.listdir(os.path.join(self.dataset_dir_made, 'trainCloth/'))))
            self.surgical = list(sorted(os.listdir(os.path.join(self.dataset_dir_made, 'trainSurgical/'))))
            self.cloth_mask = list(sorted(os.listdir(os.path.join(self.dataset_dir_made, 'cloth_mask/'))))
            self.surgical_mask = list(sorted(os.listdir(os.path.join(self.dataset_dir_made, 'surgical_mask/'))))
        self.transform_mask = transform_mask
        self.transform = transform
        self.transform_test = transform_test
    
    def __len__(self):
        if self.test:
            dir_path = os.path.join(self.dataset_dir)
        else:
            dir_path = os.path.join(self.dataset_dir_made, 'gt/')
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
            self.rand = float(random.randint(-15,15))
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
       
            gt_path = os.path.join(self.dataset_dir_made, 'gt/', self.gt[idx])
            cloth_path = os.path.join(self.dataset_dir_made, 'trainCloth/', self.cloth[idx])
            surgical_path = os.path.join(self.dataset_dir_made, 'trainSurgical/', self.surgical[idx])
            cloth_mask_path = os.path.join(self.dataset_dir_made, 'cloth_mask/', self.cloth_mask[idx])
            surgical_mask_path = os.path.join(self.dataset_dir_made, 'surgical_mask/', self.surgical_mask[idx])

            gt_img = Image.open(gt_path)
            cloth_img = Image.open(cloth_path)
            surgical_img = Image.open(surgical_path)
            cloth_mask_img = Image.open(cloth_mask_path).convert("L")
            surgical_mask_img = Image.open(surgical_mask_path).convert("L")

            gt_img = self.transform(gt_img)
            cloth_img = self.transform(cloth_img)
            surgical_img = self.transform(surgical_img)
            cloth_mask_img = self.transform(cloth_mask_img)
            surgical_mask_img = self.transform(surgical_mask_img)
            
            sample = {'WO_mask' : WOmask_img, 'W_mask' : Wmask_img, 'mask' : mask_img, 'gt' : gt_img, 'cloth' : cloth_img, 'surgical' : surgical_img, 'mask_cloth' : cloth_mask_img, 'mask_surgical' : surgical_mask_img}

        return sample
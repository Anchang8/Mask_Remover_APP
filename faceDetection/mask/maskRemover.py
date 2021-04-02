import torch
import torch.nn as nn
from torchvision import transforms, utils
import os
import time
from PIL import Image

from mask.models import Unet, Generator

# Define Weight Initialization(Spectral_Normalization and BatchNorm initialization)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.utils.spectral_norm(m)

    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def mkdir(path):
    if os.path.isdir(path) is False:
        os.mkdir(path)


def main(args):
    start = time.time()
    src = args.dst
    src = os.path.join(src, args.name)
    dst = os.path.join('cropped/after', args.name)
    # make after directory
    mkdir(dst)
    input_list = list(sorted(os.listdir(src)))

    # Transfomation PIL Image to Tensor and resize
    imsize = 512
    transform = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor(),
    ])

    # Get Pre-Trained Weights
    PreProcessing_model_dir = './mask/checkpoint/PreProcessing_checkpoint.pt'
    model_dir = './mask/checkpoint/checkpoint.pt'

    # Load the Weights from pre-trained model & Make Pre Processing Model
    PP_checkpoint = torch.load(PreProcessing_model_dir, map_location=torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'))
    preP_model_state_dict = PP_checkpoint['model_state_dict']
    preP_model = Unet()
    preP_model.apply(weights_init)
    preP_model.load_state_dict(preP_model_state_dict)
    preP_model.eval()

    # Load the Weights from pre-trained model & Make Generator
    checkpoint = torch.load(model_dir, map_location=torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'))
    generator_state_dict = checkpoint['netG_state_dict']
    generator = Generator()
    generator.apply(weights_init)
    generator.load_state_dict(generator_state_dict)
    generator.eval()

    # Get transformed result(masked face -> unmasked face)
    for i in input_list:
        if i.split('.')[-1] != 'txt':
            img_path = os.path.join(src, i)
            img = Image.open(img_path)
            img = transform(img)
            C, H, W = img.size()
            img = img.reshape((1, C, H, W))

            # Pre Processing
            pre_img = preP_model(img)

            pre_img[pre_img > 0.2] = 1
            pre_img[pre_img < 0.2] = 0

            input_img = torch.cat([img, pre_img], dim=1)
            # Generate unmasked face
            output = generator(input_img)

            img[0, 0, pre_img[0, 0, :] ==
                1] = output[0, 0, pre_img[0, 0, :] == 1]
            img[0, 1, pre_img[0, 0, :] ==
                1] = output[0, 1, pre_img[0, 0, :] == 1]
            img[0, 2, pre_img[0, 0, :] ==
                1] = output[0, 2, pre_img[0, 0, :] == 1]

            # Save the output image
            out = os.path.join(dst, i)
            utils.save_image(img, out)
        else:
            pass
    print("Runtime is {:.1f} sec".format(time.time() - start))
    return True

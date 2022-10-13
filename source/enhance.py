import argparse
import os

import matplotlib.cm as cm
import numpy as np
import skimage.color as cl
import skimage.io as io
import torch
import torchvision.utils as vutils
from torchvision.transforms import Compose, ToTensor

from models.LAPNN import LAP
from utils.image_padder import PadStride
import torch.nn.functional as F

##################################################################################################################################

parser = argparse.ArgumentParser(
    description='Laplacian Raindrop Removal - enhance')

parser.add_argument("--device", default='cpu',
                    help="device for the run [cpu cuda:-]")
parser.add_argument('--input', '-i', type=str,
                    default='', help='image to enhance')
parser.add_argument("--save_dir", '-sd', type=str,
                    default='./', help="directory where to save output")
parser.add_argument('--model', type=str,
                    default='./weights/net_coeff.pth', help='Pre-trained model path')
arser.add_argument('--levels', action='store_true',
                   help='save intermediate Laplacian levels [default false]')

opt = parser.parse_args()
# print(opt)

##################### General Options ##########################################

# directory where to save checkpoints and outputs
save_dir = opt.save_dir

# GPU board availability check
cuda_check = torch.cuda.is_available()


if not os.path.exists(save_dir):
    os.makedirs(save_dir)

############################ Network Models ####################################

with torch.no_grad():
    print("\n===> Preparing model")

    trs = Compose([
        PadStride(32, fill=-1),
        ToTensor()
    ])

    model = LAP(max_levels=3, device=opt.device)

    # Models loading
    if opt.model != '':
        model.load_state_dict(torch.load(opt.model))

    ############################ Setting cuda ######################################

    print("\n===> Setting GPU")

    if cuda_check:
        model.to(opt.device)

    print("\n===> Enhancing")

    model.eval()
    model.zero_grad()

    inputt = io.imread(opt.input)

    inputt = trs(inputt).unsqueeze(0)

    if cuda_check:
        inputt = inputt.to(opt.device)

    out, l1, l2, l3 = model(inputt)

    _, _, hh, ww = out.shape
    l2 = F.interpolate(l2, size=(hh, ww), mode='bicubic', align_corners=False)
    l3 = F.interpolate(l3, size=(hh, ww), mode='bicubic', align_corners=False)

    name = opt.input.split('/')[-1].split('.')[0]

    # Output image saving
    vutils.save_image(out, save_dir + '/' + name + '_out.png')
    if opt.levels:
        vutils.save_image(l1, save_dir + '/' + name +
                          '_l1.png', normalize=True)
        vutils.save_image(l2, save_dir + '/' + name +
                          '_l2.png', normalize=True)
        vutils.save_image(l3, save_dir + '/' + name +
                          '_l3.png', normalize=True)

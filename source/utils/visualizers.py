import random

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

def show_coeffs(coeffs):

    _, c, q, h, w = coeffs.shape

    grid = []
    for i in range(0,q):
        imgs = []
        for j in range(0,c):
            imgs.append(coeffs[:,j,i,:,:])

        grid.append(torch.cat(imgs, 2))

    out = torch.cat(grid, 1)
    out = out.squeeze(0)

    out = (out - out.min()) / (out.max() - out.min())

    return out.numpy()

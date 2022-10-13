import random
from math import floor
from os import listdir
from os.path import join

import cv2
import numpy as np
import skimage.io as io
import torch
import torch.nn.functional as F
import torch.utils.data as data
import utils.train_utils as tu
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor
from utils.image_padder import PadStride


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".JPEG", ".bmp"])


def load_img(filepath):
    img = Image.open(filepath)
    return img

################################################################################
# Training Set


class drop_train(data.Dataset):

    def __init__(self, image_dir, augm=False):
        super(drop_train, self).__init__()

        self.IN_filenames = [join(image_dir + "/train/data/", x)
                             for x in listdir(image_dir + "/train/data/") if is_image_file(x)]
        self.IN_filenames = sorted(self.IN_filenames)

        self.GT_filenames = [join(image_dir + "/train/gt/", x)
                             for x in listdir(image_dir + "/train/gt/") if is_image_file(x)]
        self.GT_filenames = sorted(self.GT_filenames)

        self.augm = augm

        self.to_tensor = Compose([
            ToTensor()
        ])

    def flip_images(self, image, target):

        if np.random.randint(0, 2) == 1:
            if np.random.randint(0, 2) == 1:
                image = image.flip(1)
                target = target.flip(1)

            else:
                image = image.flip(2)
                target = target.flip(2)

        return image, target

    def rot_images(self, image, target):

        if np.random.randint(0, 2) == 1:
            if np.random.randint(0, 2) == 1:
                image = image.transpose(2, 1)
                target = target.transpose(2, 1)
            else:
                image = image.transpose(2, 1).flip(2)
                target = target.transpose(2, 1).flip(2)

        return image, target

    def __getitem__(self, index):

        # inputt = io.imread(self.IN_filenames[index])
        inputt = Image.open(self.IN_filenames[index])
        name = self.IN_filenames[index].split("/")[-1]
        # target = io.imread(self.GT_filenames[index])
        target = Image.open(self.GT_filenames[index])

        input_g = self.to_tensor(inputt)
        target_g = self.to_tensor(target)

        if self.augm:
            [input_g, target_g] = self.flip_images(input_g, target_g)
            [input_g, target_g] = self.rot_images(input_g, target_g)

        return input_g.type(torch.FloatTensor), target_g.type(torch.FloatTensor)

    def __len__(self):
        return len(self.IN_filenames)


################################################################################
# ValidationSet


class drop_valid(data.Dataset):

    def __init__(self, image_dir):
        super(drop_valid, self).__init__()

        self.IN_filenames = [join(image_dir + "/validation/data/", x)
                             for x in listdir(image_dir + "/validation/data/") if is_image_file(x)]
        self.IN_filenames = sorted(self.IN_filenames)

        self.GT_filenames = [join(image_dir + "/validation/gt/", x)
                             for x in listdir(image_dir + "/validation/gt/") if is_image_file(x)]
        self.GT_filenames = sorted(self.GT_filenames)

        self.to_tensor = Compose([
            PadStride(32, fill=-1),
            ToTensor()
        ])

    def __getitem__(self, index):

        inputt = io.imread(self.IN_filenames[index])
        target = io.imread(self.GT_filenames[index])

        input_g = self.to_tensor(inputt)
        target_g = self.to_tensor(target)

        return input_g.type(torch.FloatTensor), target_g.type(torch.FloatTensor)

    def __len__(self):
        return len(self.IN_filenames)

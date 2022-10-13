from collections import namedtuple

import torch
from torchvision import models, transforms


def normalize_batch(batch):
    # Deprecated
    # normalize using imagenet mean and std
    mean = batch.data.new(batch.data.size())
    std = batch.data.new(batch.data.size())
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    batch = torch.div(batch, 255.0)
    batch -= mean
    batch = batch / std
    return batch

class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        # self.slice3 = torch.nn.Sequential()
        # self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 8):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(8, 16):
        #   self.slice3.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(16, 23):
        #   self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        #h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        # h = self.slice3(h)
        # h_relu3_3 = h
        # h = self.slice4(h)
        #h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu2_2'])#'relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu2_2)#h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        # self.slice3 = torch.nn.Sequential()
        # self.slice4 = torch.nn.Sequential()
        # self.slice5 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(9, 16):
        #   self.slice3.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(16, 23):
        #   self.slice4.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(23, 32):
        #   self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        #h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        # h = self.slice3(h)
        # #h_relu3_3 = h
        # h = self.slice4(h)
        # #h_relu4_3 = h
        # h = self.slice5(h)
        #h_relu5_4 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu2_2'])#['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_4'])
        out = vgg_outputs(h_relu2_2)#h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_4)
        return out

class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                # std=[0.229, 0.224, 0.225])
        self.vgg = Vgg19()
        self.crit = torch.nn.L1Loss()

    def normalize_batch(self, batch):
        # Deprecated
        # normalize using imagenet mean and std
        mean = batch.data.new(batch.data.size())
        std = batch.data.new(batch.data.size())
        mean[:, 0, :, :] = 0.485
        mean[:, 1, :, :] = 0.456
        mean[:, 2, :, :] = 0.406
        std[:, 0, :, :] = 0.229
        std[:, 1, :, :] = 0.224
        std[:, 2, :, :] = 0.225
        batch = torch.div(batch, 255.0)
        batch -= mean
        batch = batch / std
        return batch

    def forward(self, predicted, expected):

        predicted_bn = self.normalize_batch(predicted)
        expected_bn = self.normalize_batch(expected)
        predicted_features = self.vgg(predicted_bn).relu2_2
        expected_features = self.vgg(expected_bn).relu2_2

        ### Computing VGG Loss
        vgg_loss = self.crit(predicted_features, expected_features)

        return vgg_loss

    def cuda(self, device):
        self.vgg.to(device)

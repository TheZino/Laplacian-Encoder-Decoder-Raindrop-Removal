import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HuberLoss(nn.Module):
    def __init__(self, delta=.01):
        super(HuberLoss, self).__init__()
        self.delta=delta

    def __call__(self, in0, in1):
        mask = torch.zeros_like(in0)
        mann = torch.abs(in0-in1)
        eucl = .5 * (mann**2)
        mask[...] = mann < self.delta

        # loss = eucl*mask + self.delta*(mann-.5*self.delta)*(1-mask)
        loss = eucl*mask/self.delta + (mann-.5*self.delta)*(1-mask)
        return torch.sum(loss,dim=1,keepdim=True)

def gradient_loss(gen_frames, gt_frames, alpha=1):

    def gradient(x):
        # idea from tf.image.image_gradients(image)
        # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
        # x: (b,c,h,w), float32 or float64
        # dx, dy: (b,c,h,w)

        h_x = x.size()[-2]
        w_x = x.size()[-1]
        # gradient step=1
        left = x
        right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        top = x
        bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

        # dx, dy = torch.abs(right - left), torch.abs(bottom - top)
        dx, dy = right - left, bottom - top
        # dx will always have zeros in the last column, right-left
        # dy will always have zeros in the last row,    bottom-top
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0

        return dx, dy

    # gradient
    gen_dx, gen_dy = gradient(gen_frames)
    gt_dx, gt_dy = gradient(gt_frames)
    #
    grad_diff_x = (gt_dx - gen_dx)**2
    grad_diff_y = (gt_dy - gen_dy)**2

    # condense into one tensor and avg
    return torch.mean(grad_diff_x ** alpha + grad_diff_y ** alpha)

class BrightLoss(nn.Module):
    def __init__(self, lmbd=10):
        super(BrightLoss, self).__init__()
        self.lmbd = lmbd

    def __call__(self, in0, in1):

        diff = in0 - in1

        s = torch.where(diff < 0, -self.lmbd*diff, diff)

        return s.norm(1) #s.mean()

class WeightedBrightLoss(nn.Module):
    def __init__(self, lmbd=10):
        super(WeightedBrightLoss, self).__init__()
        self.lmbd = lmbd

    def __call__(self, in0, in1, map):

        diff = in0 - in1

        s = torch.where(diff < 0, -self.lmbd*diff, diff)

        s = s * map

        return s.norm(1) #s.mean()

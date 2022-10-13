import models.blocks as B
import torch
import torch.nn as nn


class PatchD70(nn.Module):

    def __init__(self, act_type='lrelu'):
        super(PatchD70, self).__init__()

        self.C64 = B.conv_block(in_nc=6, out_nc=64, \
                    kernel_size=4, stride=2, padding=1, dilation=1, \
                    bias=True, groups=1, norm_type=None, act_type=act_type)

        self.C128 = B.conv_block(in_nc=64, out_nc=128, \
                    kernel_size=4, stride=2, padding=1, dilation=1, \
                    bias=True, groups=1, norm_type='instance', act_type=act_type)

        self.C256 = B.conv_block(in_nc=128, out_nc=256, \
                    kernel_size=4, stride=2, padding=1, dilation=1, \
                    bias=True, groups=1, norm_type='instance', act_type=act_type)

        self.C512a = B.conv_block(in_nc=256, out_nc=512, \
                    kernel_size=4, stride=2, padding=1, dilation=1, \
                    bias=True, groups=1, norm_type='instance', act_type=act_type)
        self.C512b = B.conv_block(in_nc=512, out_nc=512, \
                    kernel_size=4, stride=1, padding=1, dilation=1, \
                    bias=True, groups=1, norm_type='instance', act_type=act_type)
        self.Cout = B.conv_block(in_nc=512, out_nc=1, \
                    kernel_size=4, stride=1, padding=1, dilation=1, \
                    bias=True, groups=1, norm_type=None, act_type=None)

    def forward(self, image, cond):

        x = torch.cat([image, cond], 1)

        x = self.C64(x)
        x = self.C128(x)
        x = self.C256(x)
        x = self.C512a(x)
        x = self.C512b(x)
        x = self.Cout(x)
        out = torch.sigmoid(x)

        return out

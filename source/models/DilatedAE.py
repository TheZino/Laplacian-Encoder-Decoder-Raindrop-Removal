import models.blocks as B
import torch
import torch.nn as nn

### Unet model

class DiAE(nn.Module):

    def __init__(self, in_ch=3, nf=16, act_type='lrelu'):
        super(DiAE, self).__init__()

        ### Encoding
        self.enc0  = nn.Sequential(
            B.conv_block(in_nc=in_ch, out_nc=nf, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf, out_nc=nf, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
            )

        self.enc1  = nn.Sequential(
            nn.Conv2d(in_channels = nf, out_channels = nf*2, \
                        kernel_size = 4, stride = 2, padding = 1),
            B.conv_block(in_nc=nf*2, out_nc=nf*2, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*2, out_nc=nf*2, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
            )

        ### Center
        self.central = nn.Sequential(
            nn.Conv2d(in_channels = nf*2, out_channels = nf*4, \
                        kernel_size = 4, stride = 2, padding = 1),

            B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=2, dilation=2, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=2, dilation=2, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),

            B.up_block(in_ch=nf*4, out_ch= nf*2)
            )

        ### Decoding
        self.dec1 = nn.Sequential(
            B.conv_block(in_nc=nf*2*2, out_nc=nf*2, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*2, out_nc=nf*2, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.up_block(in_ch=nf*2, out_ch=nf)
            )


        self.dec2 = nn.Sequential(
            B.conv_block(in_nc=nf*2, out_nc=nf, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf, out_nc=nf, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            )

        self.end_conv = nn.Conv2d(in_channels = nf, out_channels = in_ch, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, x):

        ## Encoder
        e1 = self.enc0(x)
        res1 = e1

        e2 = self.enc1(e1)
        res2 = e2


        ## Central
        c = self.central(e2)

        ## Decoder
        d1 = torch.cat([c, res2], 1)
        d1 = self.dec1(d1)

        d2 = torch.cat([d1, res1], 1)
        d2 = self.dec2(d2)


        out = self.end_conv(d2)

        return torch.tanh(out)

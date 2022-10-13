import models.blocks as B
import torch
import torch.nn as nn

### Unet model

class RDNet(nn.Module):

    def __init__(self, in_ch=3, nf=16, res_n = 5, act_type='lrelu'):
        super(RDNet, self).__init__()

        ### Encoding
        self.f_ext0  = nn.Sequential(
            B.conv_block(in_nc=in_ch, out_nc=nf, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=None),
            )

        self.f_ext1  = nn.Sequential(
            B.conv_block(in_nc=nf, out_nc=nf, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=None),
            )



        ### Residual dense blocks
        self.rdbs = B.make_layer(B.ResidualDenseBlock_3C(nf), res_n)

        # self.RDB1 = B.ResidualDenseBlock_3C(nf)
        # self.RDB2 = B.ResidualDenseBlock_3C(nf)
        # self.RDB3 = B.ResidualDenseBlock_3C(nf)
        # self.RDB4 = B.ResidualDenseBlock_3C(nf)
        # self.RDB5 = B.ResidualDenseBlock_3C(nf)


        ### Feature recombination
        self.f_comb = nn.Sequential(
            B.conv_block(in_nc=nf*res_n, out_nc=nf, \
                        kernel_size=1, stride=1, padding=0, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=None),
            B.conv_block(in_nc=nf, out_nc=nf, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=None),
            )

        ### Decoding
        self.dec = nn.Sequential(
            B.conv_block(in_nc=nf, out_nc=3, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=None),
            )


        self.end_conv = nn.Conv2d(in_channels = nf, out_channels = in_ch, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, x):

        idt = x
        h = self.f_ext0(x)
        grl = h
        h = self.f_ext1(h)

        ress = []
        for rdb in self.rdbs:
            h = rdb(h)
            ress.append(h)

        h = torch.cat(ress, 1)
        h = self.f_comb(h)

        h = self.dec(torch.add(h, grl))
        out = h + idt


        return out

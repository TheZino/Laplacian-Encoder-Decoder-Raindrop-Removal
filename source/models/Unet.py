import models.blocks as B
import torch
import torch.nn as nn

### Unet model

class Unet3(nn.Module):

    def __init__(self, in_ch=3, out_ch=3, nf=16, act_type='lrelu', f_act = 'sigmoid'):
        super(Unet3, self).__init__()

        if f_act == 'sigmoid':
            self.f_act = nn.Sigmoid()
        elif f_act == 'tanh':
            self.f_act = nn.Tanh()
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
            nn.MaxPool2d(kernel_size=2, stride=2),
            B.conv_block(in_nc=nf, out_nc=nf*2, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*2, out_nc=nf*2, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
            )


        self.enc2  = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            B.conv_block(in_nc=nf*2, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
            )


        ### Center
        self.central = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            B.conv_block(in_nc=nf*4, out_nc=nf*8, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*8, out_nc=nf*8, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.tconv_block(in_nc=nf*8, out_nc=nf*4, \
                        kernel_size=2, stride=2, padding=0, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
            )

        ### Decoding
        self.dec1 = nn.Sequential(
            B.conv_block(in_nc=nf*4*2, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.tconv_block(in_nc=nf*4, out_nc=nf*2, \
                        kernel_size=2, stride=2, padding=0, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
            )


        self.dec2 = nn.Sequential(
            B.conv_block(in_nc=nf*2*2, out_nc=nf*2, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*2, out_nc=nf*2, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.tconv_block(in_nc=nf*2, out_nc=nf, \
                        kernel_size=2, stride=2, padding=0, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
            )


        self.dec3 = nn.Sequential(
            B.conv_block(in_nc=nf*2, out_nc=nf, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf, out_nc=nf, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            )


        self.end_conv = nn.Conv2d(in_channels = nf, out_channels = out_ch, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, x):

        ## Encoder
        e1 = self.enc0(x)
        res1 = e1

        e2 = self.enc1(e1)
        res2 = e2

        e3 = self.enc2(e2)
        res3 = e3

        ## Central
        c = self.central(e3)

        ## Decoder
        d1 = torch.cat([c, res3], 1)
        d1 = self.dec1(d1)

        d2 = torch.cat([d1, res2], 1)
        d2 = self.dec2(d2)

        d3 = torch.cat([d2, res1], 1)
        d3 = self.dec3(d3)

        out = self.end_conv(d3)

        return self.f_act(out)





class Unet3_chk(nn.Module):

    def __init__(self, in_ch=3, out_ch=3, nf=16, act_type='lrelu', f_act = 'tanh'):
        super(Unet3_chk, self).__init__()

        if f_act == 'sigmoid':
            self.f_act = nn.Sigmoid()
        elif f_act == 'tanh':
            self.f_act = nn.Tanh()
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
            nn.MaxPool2d(kernel_size=2, stride=2),
            B.conv_block(in_nc=nf, out_nc=nf*2, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*2, out_nc=nf*2, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
            )


        self.enc2  = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            B.conv_block(in_nc=nf*2, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
            )


        ### Center
        self.central = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            B.conv_block(in_nc=nf*4, out_nc=nf*8, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*8, out_nc=nf*8, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            # B.tconv_block(in_nc=nf*8, out_nc=nf*4, \
            #             kernel_size=2, stride=2, padding=0, dilation=1, \
            #             bias=True, groups=1, norm_type=None, act_type=act_type)
            B.up_block(in_nc=nf*8, out_nc=nf*4, up_factor=2, mode='nearest')
            )

        ### Decoding
        self.dec1 = nn.Sequential(
            B.conv_block(in_nc=nf*4*2, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            # B.tconv_block(in_nc=nf*4, out_nc=nf*2, \
            #             kernel_size=2, stride=2, padding=0, dilation=1, \
            #             bias=True, groups=1, norm_type=None, act_type=act_type)
            B.up_block(in_nc=nf*4, out_nc=nf*2, up_factor=2, mode='nearest')
            )


        self.dec2 = nn.Sequential(
            B.conv_block(in_nc=nf*2*2, out_nc=nf*2, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*2, out_nc=nf*2, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            # B.tconv_block(in_nc=nf*2, out_nc=nf, \
            #             kernel_size=2, stride=2, padding=0, dilation=1, \
            #             bias=True, groups=1, norm_type=None, act_type=act_type)
            B.up_block(in_nc=nf*2, out_nc=nf, up_factor=2, mode='nearest')

            )


        self.dec3 = nn.Sequential(
            B.conv_block(in_nc=nf*2, out_nc=nf, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf, out_nc=nf, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            )


        self.end_conv = nn.Conv2d(in_channels = nf, out_channels = out_ch, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, x):

        ## Encoder
        e1 = self.enc0(x)
        res1 = e1

        e2 = self.enc1(e1)
        res2 = e2

        e3 = self.enc2(e2)
        res3 = e3

        ## Central
        c = self.central(e3)

        ## Decoder
        d1 = torch.cat([c, res3], 1)
        d1 = self.dec1(d1)

        d2 = torch.cat([d1, res2], 1)
        d2 = self.dec2(d2)

        d3 = torch.cat([d2, res1], 1)
        d3 = self.dec3(d3)

        out = self.end_conv(d3)

        return self.f_act(out)





### Aggregation Unet

class Unet3_agg(nn.Module):

    def __init__(self, in_ch=3, nf=16, act_type='relu', f_act = 'tanh'):
        super(Unet3_agg, self).__init__()

        ### Encoding
        self.enc0a = B.conv_block(in_nc=in_ch, out_nc=nf, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
        self.enc0b = B.conv_block(in_nc=nf, out_nc=nf, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
        self.dwne0a = B.conv_block(in_nc=nf, out_nc=nf, \
                        kernel_size=4, stride=2, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
        self.dwne0b = B.conv_block(in_nc=nf, out_nc=nf, \
                        kernel_size=4, stride=2, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)


        self.encagg1  = B.conv_block(in_nc=nf*2, out_nc=nf*2, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
        self.enc1  = B.conv_block(in_nc=nf*2, out_nc=nf*2, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
        self.dwne1a = B.conv_block(in_nc=nf*2, out_nc=nf*2, \
                        kernel_size=4, stride=2, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
        self.dwne1b = B.conv_block(in_nc=nf*2, out_nc=nf*2, \
                        kernel_size=4, stride=2, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)


        self.encagg2  = B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
        self.enc2  = B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
        self.dwne2a = B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=4, stride=2, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
        self.dwne2b = B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=4, stride=2, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)

        ### Center
        self.central_agg = B.conv_block(in_nc=nf*8, out_nc=nf*8, \
                    kernel_size=3, stride=1, padding=1, dilation=1, \
                    bias=True, groups=1, norm_type=None, act_type=act_type)
        self.central_aggup =  B.tconv_block(in_nc=nf*8, out_nc=nf*4, \
                                kernel_size=4, stride=2, padding=1, dilation=1, \
                                bias=True, groups=1, norm_type=None, act_type=act_type)

        self.central = nn.Sequential(
            B.conv_block(in_nc=nf*8, out_nc=nf*8, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.tconv_block(in_nc=nf*8, out_nc=nf*4, \
                        kernel_size=4, stride=2, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
            )

        ### Decoding
        self.decagg1 = B.conv_block(in_nc=nf*4*3, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
        self.dec1up = B.tconv_block(in_nc=nf*4, out_nc=nf*2,
                        kernel_size=4, stride=2, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
        self.dec1 = nn.Sequential(
                        B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                            kernel_size=3, stride=1, padding=1, dilation=1, \
                            bias=True, groups=1, norm_type=None, act_type=act_type),
                        B.tconv_block(in_nc=nf*4, out_nc=nf*2,
                            kernel_size=4, stride=2, padding=1, dilation=1, \
                            bias=True, groups=1, norm_type=None, act_type=act_type)
                    )

        self.decagg2 = B.conv_block(in_nc=nf*2*3, out_nc=nf*2, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
        self.dec2up = B.tconv_block(in_nc=nf*2, out_nc=nf, \
                        kernel_size=4, stride=2, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
        self.dec2 = nn.Sequential(
                        B.conv_block(in_nc=nf*2, out_nc=nf*2, \
                            kernel_size=3, stride=1, padding=1, dilation=1, \
                            bias=True, groups=1, norm_type=None, act_type=act_type),
                        B.tconv_block(in_nc=nf*2, out_nc=nf, \
                            kernel_size=4, stride=2, padding=1, dilation=1, \
                            bias=True, groups=1, norm_type=None, act_type=act_type)
                    )

        self.dec3 = nn.Sequential(
                        B.conv_block(in_nc=nf*3, out_nc=nf, \
                            kernel_size=3, stride=1, padding=1, dilation=1, \
                            bias=True, groups=1, norm_type=None, act_type=act_type),
                        B.conv_block(in_nc=nf, out_nc=3, \
                            kernel_size=3, stride=1, padding=1, dilation=1, \
                            bias=True, groups=1, norm_type=None, act_type=act_type)
                    )

    def forward(self, x):

        ## Encoder
        e0a = self.enc0a(x)
        e0b = self.enc0b(e0a)
        res1 = e0b
        e0a = self.dwne0a(e0a)
        e0b = self.dwne0b(e0b)


        e1a = self.encagg1(torch.cat([e0a, e0b], 1))
        e1b = self.enc1(e1a)
        res2 = e1b
        e1a = self.dwne1a(e1a)
        e1b = self.dwne1b(e1b)


        e2a = self.encagg2(torch.cat([e1a, e1b], 1))
        e2b = self.enc2(e2a)
        res3 = e2b
        e2a = self.dwne2a(e2a)
        e2b = self.dwne2b(e2b)

        ## Central
        ca = self.central_agg(torch.cat([e2a, e2b], 1))
        cb = self.central(ca)
        ca = self.central_aggup(ca)

        ## Decoder
        d1a = self.decagg1(torch.cat([ca, cb, res3], 1))
        d1b = self.dec1(d1a)
        d1a = self.dec1up(d1a)

        d2a = self.decagg2(torch.cat([d1a, d1b, res2], 1))
        d2b = self.dec2(d2a)
        d2a = self.dec2up(d2a)

        d3 = self.dec3(torch.cat([d2a, d2b, res1], 1))

        return torch.tanh(d3)




##### NEW AE

class AE_net(nn.Module):

    def __init__(self, in_ch=3, out_ch=3, nf=16, act_type='lrelu', f_act = 'tanh'):
        super(AE_net, self).__init__()

        if f_act == 'sigmoid':
            self.f_act = nn.Sigmoid()
        elif f_act == 'tanh':
            self.f_act = nn.Tanh()
        ### Encoding
        self.enc0  = nn.Sequential(
            B.conv_block(in_nc=in_ch, out_nc=nf, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
            )

        self.enc1  = nn.Sequential(
            # nn.MaxPool2d(kernel_size=2, stride=2),
            B.conv_block(in_nc=nf, out_nc=nf, \
                        kernel_size=3, stride=2, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf, out_nc=nf*2, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),

            )


        # self.enc2  = nn.Sequential(
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     B.conv_block(in_nc=nf*2, out_nc=nf*4, \
        #                 kernel_size=3, stride=1, padding=1, dilation=1, \
        #                 bias=True, groups=1, norm_type=None, act_type=act_type),
        #     B.conv_block(in_nc=nf*4, out_nc=nf*4, \
        #                 kernel_size=3, stride=1, padding=1, dilation=1, \
        #                 bias=True, groups=1, norm_type=None, act_type=act_type)
        #     )


        ### Center
        self.central = nn.Sequential(
            # nn.MaxPool2d(kernel_size=2, stride=2),
            B.conv_block(in_nc=nf*2, out_nc=nf*2, \
                        kernel_size=3, stride=2, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),

            B.conv_block(in_nc=nf*2, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),

            B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=2, dilation=2, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=4, dilation=4, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=8, dilation=8, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=16, dilation=16, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),

            B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.tconv_block(in_nc=nf*4, out_nc=nf*2, \
                        kernel_size=2, stride=2, padding=0, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type='relu')

            # B.up_block(in_nc=nf*4, out_nc=nf*2, up_factor=2, mode='nearest')
            )

        ### Decoding
        # self.dec1 = nn.Sequential(
        #     B.conv_block(in_nc=nf*4*2, out_nc=nf*4, \
        #                 kernel_size=3, stride=1, padding=1, dilation=1, \
        #                 bias=True, groups=1, norm_type=None, act_type=act_type),
        #     # B.conv_block(in_nc=nf*4, out_nc=nf*4, \
        #     #             kernel_size=3, stride=1, padding=1, dilation=1, \
        #     #             bias=True, groups=1, norm_type=None, act_type=act_type),
        #     # B.tconv_block(in_nc=nf*4, out_nc=nf*2, \
        #     #             kernel_size=2, stride=2, padding=0, dilation=1, \
        #     #             bias=True, groups=1, norm_type=None, act_type=act_type)
        #     B.up_block(in_nc=nf*4, out_nc=nf*2, up_factor=2, mode='nearest')
        #     )


        self.dec2 = nn.Sequential(
            B.conv_block(in_nc=nf*2*2, out_nc=nf*2, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            # B.conv_block(in_nc=nf*2, out_nc=nf*2, \
            #             kernel_size=3, stride=1, padding=1, dilation=1, \
            #             bias=True, groups=1, norm_type=None, act_type=act_type),
            B.tconv_block(in_nc=nf*2, out_nc=nf, \
                        kernel_size=2, stride=2, padding=0, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type='relu')
            # B.up_block(in_nc=nf*2, out_nc=nf, up_factor=2, mode='nearest')

            )


        self.dec3 = nn.Sequential(
            B.conv_block(in_nc=nf*2, out_nc=nf, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf, out_nc=nf, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            )


        self.end_conv = nn.Conv2d(in_channels = nf, out_channels = out_ch, kernel_size = 3, stride = 1, padding = 1)

    def forward(self, x):

        ## Encoder
        e1 = self.enc0(x)
        res1 = e1

        e2 = self.enc1(e1)
        res2 = e2

        # e3 = self.enc2(e2)
        # res3 = e3

        ## Central
        c = self.central(e2)

        ## Decoder
        # d1 = torch.cat([c, res3], 1)
        # d1 = self.dec1(d1)

        d2 = torch.cat([c, res2], 1)
        d2 = self.dec2(d2)

        d3 = torch.cat([d2, res1], 1)
        d3 = self.dec3(d3)

        out = self.end_conv(d3)

        return self.f_act(out)

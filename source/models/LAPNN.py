import models.blocks as B
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils.train_utils as tu
from models.DilatedAE import DiAE
from models.RDNet import RDNet
from models.Unet import AE_net, Unet3, Unet3_agg, Unet3_chk
from utils import colors as cl
from utils.weight_initializers import init_weights


def gauss_kernel(size=5, device=torch.device('cpu'), channels=3):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel

def conv_gauss(img, kernel):
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
    return out

def laplacian_pyramid(img, kernel, max_levels=3):
    current = img
    pyr = []
    for level in range(max_levels-1):
        filtered = conv_gauss(current, kernel)
        down = filtered[:, :, ::2, ::2]
        up = F.interpolate(down, size=(current.shape[2], current.shape[3]), mode='bicubic', align_corners=False)
        diff = current-up
        pyr.append(diff)
        current = down
    pyr.append(current)
    return pyr

def recomposition(pyr):
    out = pyr[0]
    for lvl in pyr[1:]:
        _,_,hh,ww = lvl.shape
        out = F.interpolate(out, size=(hh, ww), mode='bicubic', align_corners=False)
        out = out + lvl

    return out



### Unet model

class LAP(nn.Module):

    def __init__(self, max_levels=3, channels=3, device=torch.device('cpu')):
        super(LAP, self).__init__()

        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels, device=device)

        self.CLVL1 = Unet3_chk(in_ch=channels, nf=24)
        self.CLVL2 = Unet3_chk(in_ch=channels, nf=32)
        self.CLVL3 = Unet3_chk(in_ch=channels, nf=64, f_act='sigmoid')

        init_weights(self.CLVL1, init_type='kaiming')
        init_weights(self.CLVL2, init_type='kaiming')
        init_weights(self.CLVL3, init_type='kaiming')


    def get_optimizers(self, opt):

        self.opt1 = optim.Adam(self.CLVL1.parameters(),
                                    lr=opt.lr,
                                    betas=(opt.beta, 0.999),
                                    weight_decay=1e-8
                                    )
        self.opt2 = optim.Adam(self.CLVL2.parameters(),
                                    lr=opt.lr,
                                    betas=(opt.beta, 0.999),
                                    weight_decay=1e-8
                                    )
        self.opt3 = optim.Adam(self.CLVL3.parameters(),
                                    lr=opt.lr,
                                    betas=(opt.beta, 0.999),
                                    weight_decay=1e-8
                                    )

        return [self.opt1, self.opt2, self.opt3]

    def load_chkp(self, path, epoch, test=False):

        name = 'LAPNN'

        if epoch == 'last':
            path = path + "/last_ckp/"
        else:
            path = path + "/checkpoints/"

        tu.load_model(self.CLVL1, path + '/net' + name + '_LVL1_epoch' + str(epoch) +'.pth')
        tu.load_model(self.CLVL2, path + '/net' + name + '_LVL2_epoch' + str(epoch) +'.pth')
        tu.load_model(self.CLVL3, path + '/net' + name + '_LVL3_epoch' + str(epoch) +'.pth')

        if not test:
            tu.load_optimizer(self.opt1, path + '/optim' + name + '_epoch' + str(epoch) +'_LVL1.pth')
            tu.load_optimizer(self.opt2, path + '/optim' + name + '_epoch' + str(epoch) +'_LVL2.pth')
            tu.load_optimizer(self.opt3, path + '/optim' + name + '_epoch' + str(epoch) +'_LVL3.pth')

    def forward(self, x):

        #### Input must be an image in range [0,1]

        # [l1_lp, l2_lp, l3_lp] = laplacian_pyramid(x, self.gauss_kernel, max_levels=self.max_levels)

        _,_,hh,ww = x.shape
        l2 = F.interpolate(x, size=(hh/2, ww/2), mode='bicubic', align_corners=False)
        l3 = F.interpolate(x, size=(hh/4, ww/4), mode='bicubic', align_corners=False)


        ### cnn lvl1
        l1_n = self.CLVL1(x)
        # l1_n = l1
        ### cnn lvl2
        l2_n = self.CLVL2(l2)
        # l2_n = l2
        ### cnn lvl3
        l3_n = self.CLVL3(l3)

        out = recomposition([l3_n, l2_n, l1_n])

        return out, l1_n, l2_n, l3_n

    def save(self, check_path, epoch):
        name = "LAPNN"
        tu.save_model(self.CLVL1.state_dict(), check_path, epoch, name+'_LVL1')
        tu.save_model(self.CLVL2.state_dict(), check_path, epoch, name+'_LVL2')
        tu.save_model(self.CLVL3.state_dict(), check_path, epoch, name+'_LVL3')

        tu.save_optim(self.opt1.state_dict(), check_path, epoch, name + '_opt1')
        tu.save_optim(self.opt2.state_dict(), check_path, epoch, name + '_opt2')
        tu.save_optim(self.opt3.state_dict(), check_path, epoch, name + '_opt3')

        tu.save_model(self.CLVL1.state_dict(), check_path + '/../last_ckp', 'last', name+'_LVL1')
        tu.save_model(self.CLVL2.state_dict(), check_path + '/../last_ckp', 'last', name+'_LVL2')
        tu.save_model(self.CLVL3.state_dict(), check_path + '/../last_ckp', 'last', name+'_LVL3')

        tu.save_optim(self.opt1.state_dict(), check_path + '/../last_ckp', epoch, name + '_opt1')
        tu.save_optim(self.opt2.state_dict(), check_path + '/../last_ckp', epoch, name + '_opt2')
        tu.save_optim(self.opt3.state_dict(), check_path + '/../last_ckp', epoch, name + '_opt3')





class LAP_one(nn.Module):

    def __init__(self, max_levels=3, channels=3, device=torch.device('cpu')):
        super(LAP_one, self).__init__()

        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels, device=device)


        self.CLVL3 = Unet3(in_ch=channels, nf=64, f_act='sigmoid')

        init_weights(self.CLVL3, init_type='kaiming')


    def get_optimizers(self, opt):

        self.opt3 = optim.Adam(self.CLVL3.parameters(),
                                    lr=opt.lr,
                                    betas=(opt.beta, 0.999)
                                    # weight_decay=1e-8
                                    )

        return [self.opt3]

    def load_chkp(self, path, epoch, test=False):

        name = 'LAPNN'

        if epoch == 'last':
            path = path + "/last_ckp/"
        else:
            path = path + "/checkpoints/"

        tu.load_model(self.CLVL3, path + '/net' + name + '_LVL3_epoch' + str(epoch) +'.pth')

        if not test:
            tu.load_optimizer(self.opt3, path + '/optim' + name + '_epoch' + str(epoch) +'_LVL3.pth')

    def forward(self, x):

        #### Input must be an image in range [0,1]

        out = self.CLVL3(x)

        return out

    def save(self, check_path, epoch):
        name = "LAPNN"
        tu.save_model(self.CLVL3.state_dict(), check_path, epoch, name+'_LVL3')
        tu.save_optim(self.opt3.state_dict(), check_path, epoch, name + '_opt3')

        tu.save_model(self.CLVL3.state_dict(), check_path + '/../last_ckp', 'last', name+'_LVL3')

        tu.save_optim(self.opt3.state_dict(), check_path + '/../last_ckp', epoch, name + '_opt3')



#### Ycbcr

class LAP_ycbcr(nn.Module):

    def __init__(self, max_levels=3, channels=1, device=torch.device('cpu')):
        super(LAP_ycbcr, self).__init__()

        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels, device=device)

        self.CLVL1 = Unet3_chk(in_ch=channels, out_ch=channels, nf=24)
        self.CLVL2 = Unet3_chk(in_ch=channels, out_ch=channels, nf=32)
        self.CLVL3 = AE_net(in_ch=channels, out_ch=channels, nf=64, f_act='sigmoid')

        self.CColor = Unet3_chk(in_ch=3, out_ch=2, nf=32, f_act='sigmoid')

        init_weights(self.CLVL1, init_type='kaiming')
        init_weights(self.CLVL2, init_type='kaiming')
        init_weights(self.CLVL3, init_type='kaiming')
        # init_weights(self.CColor, init_type='kaiming')


    def get_optimizers(self, opt):

        self.opt1 = optim.Adam(self.CLVL1.parameters(),
                                    lr=opt.lr,
                                    betas=(opt.beta, 0.999),
                                    weight_decay=1e-8
                                    )
        self.opt2 = optim.Adam(self.CLVL2.parameters(),
                                    lr=opt.lr,
                                    betas=(opt.beta, 0.999),
                                    weight_decay=1e-8
                                    )
        self.opt3 = optim.Adam(self.CLVL3.parameters(),
                                    lr=opt.lr,
                                    betas=(opt.beta, 0.999),
                                    weight_decay=1e-8
                                    )
        # self.opt4 = optim.Adam(self.CColor.parameters(),
        #                             lr=opt.lr,
        #                             betas=(opt.beta, 0.999),
        #                             weight_decay=1e-8
        #                             )

        return [self.opt1, self.opt2, self.opt3]#, self.opt4]

    def load_chkp(self, path, epoch, test=False):

        name = 'LAPNN'

        if epoch == 'last':
            path = path + "/last_ckp/"
        else:
            path = path + "/checkpoints/"

        tu.load_model(self.CLVL1, path + '/net' + name + '_LVL1_epoch' + str(epoch) +'.pth')
        tu.load_model(self.CLVL2, path + '/net' + name + '_LVL2_epoch' + str(epoch) +'.pth')
        tu.load_model(self.CLVL3, path + '/net' + name + '_LVL3_epoch' + str(epoch) +'.pth')
        # tu.load_model(self.CColor, path + '/net' + name + '_Color_epoch' + str(epoch) +'.pth')

        if not test:
            tu.load_optimizer(self.opt1, path + '/optim' + name + '_epoch' + str(epoch) +'_LVL1.pth')
            tu.load_optimizer(self.opt2, path + '/optim' + name + '_epoch' + str(epoch) +'_LVL2.pth')
            tu.load_optimizer(self.opt3, path + '/optim' + name + '_epoch' + str(epoch) +'_LVL3.pth')

    def forward(self, x):

        #### Input must be an image in range [0,1]

        x = cl.rgb2ycbcr(x)

        y_in = x[:,0,:,:].unsqueeze(1)
        c_in = x[:,1:,:,:]

        [l1, l2, l3] = laplacian_pyramid(y_in, self.gauss_kernel, max_levels=self.max_levels)

        ### cnn lvl1
        l1_n = self.CLVL1(l1)
        # l1_n = l1
        ### cnn lvl2
        l2_n = self.CLVL2(l2)
        # l2_n = l2
        ### cnn lvl3
        l3_n = self.CLVL3(l3)

        y_out = recomposition([l3_n, l2_n, l1_n])
        # c_out = self.CColor(torch.cat([y_out.detach(), c_in], 1))


        out = cl.ycbcr2rgb(torch.cat([y_out, c_in], 1))

        return out, y_out, l1_n, l2_n, l3_n, c_in

    def save(self, check_path, epoch):
        name = "LAPNN"
        tu.save_model(self.CLVL1.state_dict(), check_path, epoch, name+'_LVL1')
        tu.save_model(self.CLVL2.state_dict(), check_path, epoch, name+'_LVL2')
        tu.save_model(self.CLVL3.state_dict(), check_path, epoch, name+'_LVL3')
        tu.save_model(self.CColor.state_dict(), check_path, epoch, name+'_Color')

        tu.save_model(self.CLVL1.state_dict(), check_path + '/../last_ckp', 'last', name+'_LVL1')
        tu.save_model(self.CLVL2.state_dict(), check_path + '/../last_ckp', 'last', name+'_LVL2')
        tu.save_model(self.CLVL3.state_dict(), check_path + '/../last_ckp', 'last', name+'_LVL3')
        tu.save_model(self.CColor.state_dict(), check_path + '/../last_ckp', 'last', name+'_Color')













### Training LVL2 with trained LVL3

class LAPl2(nn.Module):

    def __init__(self, max_levels=3, channels=3, device=torch.device('cpu')):
        super(LAPl2, self).__init__()

        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels, device=device)

        self.CLVL2 = RDNet(nf = 32)
        self.CLVL3 = DiAE(nf = 64)

        init_weights(self.CLVL2, init_type='kaiming_small')


    def load_chpt(path):

        tu.load_model(self.CLVL3, path)
        for param in self.CLVL3.parameters():
            param.requires_grad = False

    def get_optimizers(self, opt):

        opt2 = optim.Adam(self.CLVL2.parameters(),
                                    lr=opt.lr,
                                    betas=(opt.beta, 0.999),
                                    weight_decay=1e-8
                                    )

        return [opt2]

    def forward(self, x):

        [l1, l2, l3] = laplacian_pyramid(x, self.gauss_kernel, max_levels=self.max_levels)

        ### cnn lvl1
        l1_n = l1
        ### cnn lvl2
        l2_n = self.CLVL2(l2)
        ### cnn lvl3
        with torch.nograd():
            l3_n = self.CLVL3(l3)

        out = recomposition([l3_n, l2_n, l1_n])


        return out, l1_n, l2_n, l3_n

    def save(self, check_path, epoch):
        name = "LAPNN"
        tu.save_model(self.CLVL2.state_dict(), check_path, epoch, name+'_LVL2')
        tu.save_model(self.CLVL2.state_dict(), check_path + '/../last_ckp', 'last', name+'_LVL2')


class LAPl3(nn.Module):

    def __init__(self, max_levels=3, channels=3, device=torch.device('cpu')):
        super(LAPl3, self).__init__()

        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels, device=device)

        self.CLVL3 = DiAE(nf = 64)

        init_weights(self.CLVL3, init_type='kaiming_small')


    def get_optimizers(self, opt):

        opt3 = optim.Adam(self.CLVL3.parameters(),
                                    lr=opt.lr,
                                    betas=(opt.beta, 0.999),
                                    weight_decay=1e-8
                                    )

        return [opt3]

    def forward(self, x):

        [l1, l2, l3] = laplacian_pyramid(x, self.gauss_kernel, max_levels=self.max_levels)

        ### cnn lvl1
        l1_n = l1
        ### cnn lvl2
        l2_n = l2
        ### cnn lvl3
        l3_n = self.CLVL3(l3)

        out = recomposition([l3_n, l2_n, l1_n])


        return out, l3_n

    def save(self, check_path, epoch):
        name = "LAPNN"
        tu.save_model(self.CLVL3.state_dict(), check_path, epoch, name+'_LVL3')
        tu.save_model(self.CLVL3.state_dict(), check_path + '/../last_ckp', 'last', name+'_LVL3')




#### New Version


class LAP_v2(nn.Module):

    def __init__(self, in_ch=3, out_ch=3, nf=16, act_type='lrelu', f_act = 'sigmoid'):
        super(LAP_v2, self).__init__()

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
        self.rdb_l1 = B.make_layer(B.RRDB(nf), 2)

        self.enc1  = nn.Sequential(
            B.conv_block(in_nc=nf, out_nc=nf*2, \
                        kernel_size=3, stride=2, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*2, out_nc=nf*2, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),

            )
        self.rdb_l2 = B.make_layer(B.RRDB(nf*2), 2)

        ### Center
        self.rdb_l3 = nn.Sequential(
            # nn.MaxPool2d(kernel_size=2, stride=2),
            B.conv_block(in_nc=nf*2, out_nc=nf*4, \
                        kernel_size=3, stride=2, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),

            B.RRDB(nc=nf*4, kernel_size=3, gc=32, stride=1, bias=True),
            B.RRDB(nc=nf*4, kernel_size=3, gc=32, stride=1, bias=True),
            # B.RRDB(nc=nf*4, kernel_size=3, gc=32, stride=1, bias=True),

            B.tconv_block(in_nc=nf*4, out_nc=3, \
                        kernel_size=2, stride=2, padding=0, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type='relu')
            )


        self.out_lvl2 = B.conv_block(in_nc=nf*2, out_nc=3, \
                            kernel_size=3, stride=1, padding=1, dilation=1, \
                            bias=True, groups=1, norm_type=None, act_type=act_type)
        self.up_lvl2 = B.tconv_block(in_nc=3, out_nc=3, \
                            kernel_size=2, stride=2, padding=0, dilation=1, \
                            bias=True, groups=1, norm_type=None, act_type='relu')
        self.out_lvl1 = B.conv_block(in_nc=nf, out_nc=3, \
                            kernel_size=3, stride=1, padding=1, dilation=1, \
                            bias=True, groups=1, norm_type=None, act_type=act_type)

    def forward(self, x):

        l1 = self.enc0(x)
        l1_n = self.rdb_l1(l1)

        l2 = self.enc1(l1)
        l2_n = self.rdb_l2(l2)

        l3_n = self.rdb_l3(l2)

        l2_n = torch.add(l3_n, self.out_lvl2(l2_n))
        l2_n = self.up_lvl2(l2_n)

        out = torch.add(l2_n, self.out_lvl1(l1_n))


        return self.f_act(out)








class LAP_v3(nn.Module):

    def __init__(self, max_levels=3, channels=3, device=torch.device('cpu')):
        super(LAP_v3, self).__init__()

        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels, device=device)

        self.CLVL3 = Unet3(in_ch=channels, nf=64)
        # nn.Sequential(
        #                 B.conv_block(in_nc=3, out_nc=32, \
        #                             kernel_size=3, stride=1, padding=1, dilation=1, \
        #                             bias=True, groups=1, norm_type=None, act_type='lrelu'),
        #                 B.ResidualDenseBlock_5C(nc=32, kernel_size=3, gc=32, stride=1, bias=True),
        #                 B.conv_block(in_nc=32, out_nc=3, \
        #                             kernel_size=3, stride=1, padding=1, dilation=1, \
        #                             bias=False, groups=1, norm_type=None, act_type=None)
        #             )
        self.CLVL2 = Unet3(in_ch=channels, nf=32)
        # nn.Sequential(
        #                 B.conv_block(in_nc=3, out_nc=32, \
        #                             kernel_size=3, stride=1, padding=1, dilation=1, \
        #                             bias=True, groups=1, norm_type=None, act_type='lrelu'),
        #                 B.ResidualDenseBlock_5C(nc=32, kernel_size=3, gc=32, stride=1, bias=True),
        #                 B.conv_block(in_nc=32, out_nc=3, \
        #                             kernel_size=3, stride=1, padding=1, dilation=1, \
        #                             bias=False, groups=1, norm_type=None, act_type=None)
        #             )

        self.CLVL1 = Unet3(in_ch=channels, nf=32)
        # nn.Sequential(
        #                 B.conv_block(in_nc=3, out_nc=32, \
        #                             kernel_size=3, stride=1, padding=1, dilation=1, \
        #                             bias=True, groups=1, norm_type=None, act_type='lrelu'),
        #                 B.ResidualDenseBlock_5C(nc=32, kernel_size=3, gc=32, stride=1, bias=True),
        #                 B.conv_block(in_nc=32, out_nc=3, \
        #                             kernel_size=3, stride=1, padding=1, dilation=1, \
        #                             bias=False, groups=1, norm_type=None, act_type=None)
        #             )

        # self.up_lvl3 = B.tconv_block(in_nc=3, out_nc=3, \
        #                     kernel_size=4, stride=2, padding=1, dilation=1, \
        #                     bias=True, groups=1, norm_type=None, act_type='relu')
        # self.up_lvl2 = B.tconv_block(in_nc=3, out_nc=3, \
        #                     kernel_size=4, stride=2, padding=1, dilation=1, \
        #                     bias=True, groups=1, norm_type=None, act_type='relu')


    def forward(self, x):

        #### Input must be an image in range [0,1]

        [l1, l2, l3] = laplacian_pyramid(x, self.gauss_kernel, max_levels=self.max_levels)


        ### cnn lvl1
        l3_n = self.CLVL3(l3)
        l3_n = torch.sigmoid(l3_n)
        l3_up = F.interpolate(l3_n, size=(l2.shape[2], l2.shape[3]), mode='bicubic', align_corners=False)
        # l3_up = self.up_lvl3(l3_n)
        # l3_up = torch.sigmoid(l3_up)

        ### cnn lvl2
        l2_in = torch.add(l3_up,l2)
        l2_n = self.CLVL2(l2_in)
        l2_n = torch.add(l2_n, l2_in)
        l2_up = F.interpolate(l2_n, size=(l1.shape[2], l1.shape[3]), mode='bicubic', align_corners=False)
        # l2_up = self.up_lvl2(l2_n)
        # l2_up = torch.sigmoid(l2_up)

        ### cnn lvl3
        l1_in = torch.add(l2_up,l1)
        l1_n = self.CLVL1(l1_in)
        l1_n = torch.add(l1_n, l1_in)


        # out = recomposition([l3_n, l2_n, l1_n])

        return l1_n, l2_n, l3_n


















class LAP_v4(nn.Module):

    def __init__(self, nf=32, act_type='lrelu', device=torch.device('cpu')):
        super(LAP_v4, self).__init__()

        self.gauss_kernel = gauss_kernel(channels=3, device=device)

        self.enc0  = nn.Sequential(
            B.conv_block(in_nc=3, out_nc=nf, \
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
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=2, dilation=2, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=4, dilation=4, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
            )
        # self.finalmp = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dec_l3 = nn.Sequential(

            B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*4, out_nc=nf*2, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),

            B.conv_block(in_nc=nf*2, out_nc=3, \
                        kernel_size=1, stride=1, padding=0, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=None)
            )

        self.dec_l2_0 = nn.Sequential(

            B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.tconv_block(in_nc=nf*4, out_nc=nf*2, \
                        kernel_size=2, stride=2, padding=0, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
            )

        self.dec_l2_1 = nn.Sequential(

            B.conv_block(in_nc=nf*2, out_nc=nf*2, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*2, out_nc=nf*2, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*2, out_nc=nf*2, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*2, out_nc=nf*2, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*2, out_nc=3, \
                        kernel_size=1, stride=1, padding=0, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=None)
            )

        self.dec_l1_0 = nn.Sequential(

            B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.tconv_block(in_nc=nf*4, out_nc=nf*2, \
                        kernel_size=2, stride=2, padding=0, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
            )

        self.dec_l1_1 = nn.Sequential(

            B.conv_block(in_nc=nf*2, out_nc=nf*2, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*2, out_nc=nf*2, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.tconv_block(in_nc=nf*2, out_nc=nf, \
                        kernel_size=2, stride=2, padding=0, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
            )

        self.dec_l1_2 = nn.Sequential(

            B.conv_block(in_nc=nf, out_nc=nf, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf, out_nc=nf, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf, out_nc=3, \
                        kernel_size=1, stride=1, padding=0, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=None)
            )


    def encode(self, x):

        # import ipdb; ipdb.set_trace()
        e1 = self.enc0(x)
        res1 = e1

        e2 = self.enc1(e1)
        res2 = e2

        e3 = self.enc2(e2)
        res3 = e3

        # feat = self.finalmp(e3)
        feat = e3

        return feat, res1, res2

    def decode_l3(self, x):

        return self.dec_l3(x)

    def decode_l2(self, x, res):

        # import ipdb; ipdb.set_trace()
        x = self.dec_l2_0(x)

        x = self.dec_l2_1(torch.add(x, res))

        return x

    def decode_l1(self, x, res2, res1):

        x = self.dec_l1_0(x)

        x = self.dec_l1_1(torch.add(x, res2))

        x = self.dec_l1_2(torch.add(x, res1))

        return x

    def forward(self, x):

        # _,_,l3_p = laplacian_pyramid(x, self.gauss_kernel, max_levels=3)


        feat, res1, res2 = self.encode(x)

        l3_n = self.decode_l3(feat)
        l3_n = torch.sigmoid(l3_n)
        l2_n = self.decode_l2(feat, res2)
        l2_n = torch.tanh(l2_n)
        l1_n = self.decode_l1(feat, res2, res1)
        l1_n = torch.tanh(l1_n)

        # out = recomposition([torch.add(l3_n,l3_p), l2_n, l1_n])
        out = recomposition([l3_n, l2_n, l1_n])

        return out, l1_n, l2_n, l3_n



class LAP_v4_1(nn.Module):

    def __init__(self, nf=32, act_type='lrelu', device=torch.device('cpu')):
        super(LAP_v4_1, self).__init__()

        self.gauss_kernel = gauss_kernel(channels=3, device=device)

        self.enc0  = nn.Sequential(
            B.conv_block(in_nc=3, out_nc=nf, \
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
                        bias=True, groups=1, norm_type=None, act_type=act_type)
            )

        self.dec_l3 = nn.Sequential(

            B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*4, out_nc=nf*2, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),

            B.conv_block(in_nc=nf*2, out_nc=3, \
                        kernel_size=1, stride=1, padding=0, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=None)
            )

        self.dec_0 = nn.Sequential(

            B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.tconv_block(in_nc=nf*4, out_nc=nf*2, \
                        kernel_size=2, stride=2, padding=0, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
            )

        self.dec_1 = nn.Sequential(
            B.conv_block(in_nc=nf*2, out_nc=nf*2, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*2, out_nc=nf*2, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*2, out_nc=nf*2, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*2, out_nc=nf*2, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            )

        self.dec_l2_out = B.conv_block(in_nc=nf*2, out_nc=3, \
                            kernel_size=1, stride=1, padding=0, dilation=1, \
                            bias=True, groups=1, norm_type=None, act_type=None)

        self.dec_up_l21 = B.tconv_block(in_nc=nf*2, out_nc=nf, \
                            kernel_size=2, stride=2, padding=0, dilation=1, \
                            bias=True, groups=1, norm_type=None, act_type=act_type)

        self.dec_l1_out = nn.Sequential(
            B.conv_block(in_nc=nf, out_nc=nf, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf, out_nc=nf, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf, out_nc=nf, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf, out_nc=nf, \
                        kernel_size=3, stride=1, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf, out_nc=3, \
                        kernel_size=1, stride=1, padding=0, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=None)
            )


    def encode(self, x):

        e1 = self.enc0(x)
        res1 = e1

        e2 = self.enc1(e1)
        res2 = e2

        e3 = self.enc2(e2)
        res3 = e3

        feat = e3

        return feat, res1, res2

    def decode_l3(self, x):

        return self.dec_l3(x)

    def decode_l21(self, x, res2, res1):

        # import ipdb; ipdb.set_trace()
        x = self.dec_0(x)

        x = self.dec_1(torch.add(x, res2))

        l2 = self.dec_l2_out(x)

        x = self.dec_up_l21(x)

        l1 = self.dec_l1_out(torch.add(x, res1))

        return l2, l1

    def forward(self, x):

        feat, res1, res2 = self.encode(x)

        l3_n = self.decode_l3(feat)
        l3_n = torch.sigmoid(l3_n)
        l2_n, l1_n = self.decode_l21(feat, res2, res1)
        l2_n = torch.tanh(l2_n)
        l1_n = torch.tanh(l1_n)

        out = recomposition([l3_n, l2_n, l1_n])

        return out, l1_n, l2_n, l3_n









class LAP_v4_2(nn.Module):

    def __init__(self, nf=32, act_type='relu', device=torch.device('cpu')):
        super(LAP_v4_2, self).__init__()

        self.gauss_kernel = gauss_kernel(channels=3, device=device)

        self.enc0  = B.conv_block(in_nc=3, out_nc=nf, \
                        kernel_size=5, stride=1, padding=2, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)

        self.enc1_1  = B.ResBlock_att(in_nc=nf, out_nc=nf, \
                        kernel_size=5, stride=1, padding=2, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
        self.enc1_2  = B.ResBlock_att(in_nc=nf, out_nc=nf, \
                        kernel_size=5, stride=1, padding=2, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
        self.enc1_3  = B.ResBlock_att(in_nc=nf, out_nc=nf, \
                        kernel_size=5, stride=1, padding=2, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)

        self.enc2_0 = B.conv_block(in_nc=nf, out_nc=nf*2, \
                        kernel_size=3, stride=2, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
        self.enc2_1  = B.ResBlock_att(in_nc=nf*2, out_nc=nf*2, \
                        kernel_size=5, stride=1, padding=2, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
        self.enc2_2  = B.ResBlock_att(in_nc=nf*2, out_nc=nf*2, \
                        kernel_size=5, stride=1, padding=2, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
        self.enc2_3  = B.ResBlock_att(in_nc=nf*2, out_nc=nf*2, \
                        kernel_size=5, stride=1, padding=2, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)

        self.enc3_0 = B.conv_block(in_nc=nf*2, out_nc=nf*4, \
                        kernel_size=3, stride=2, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
        self.enc3_1  = B.ResBlock_att(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=5, stride=1, padding=2, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
        self.enc3_2  = B.ResBlock_att(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=5, stride=1, padding=2, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
        self.enc3_3  = B.ResBlock_att(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=5, stride=1, padding=2, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)

        ########################################################################


        self.dec_l3_0 = B.ResBlock_att(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=5, stride=1, padding=2, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
        self.dec_l3_1 = B.ResBlock_att(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=5, stride=1, padding=2, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
        self.dec_l3_2 = B.ResBlock_att(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=5, stride=1, padding=2, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
        self.dec_l3_3 = B.conv_block(in_nc=nf*4, out_nc=3, \
                        kernel_size=1, stride=1, padding=0, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=None)

        self.dec_0 = nn.Sequential(

            B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=5, stride=1, padding=2, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*4, out_nc=nf*4, \
                        kernel_size=5, stride=1, padding=2, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.tconv_block(in_nc=nf*4, out_nc=nf*2, \
                        kernel_size=4, stride=2, padding=1, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type)
            )

        self.dec_1 = nn.Sequential(
            B.conv_block(in_nc=nf*2, out_nc=nf*2, \
                        kernel_size=5, stride=1, padding=2, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*2, out_nc=nf*2, \
                        kernel_size=5, stride=1, padding=2, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*2, out_nc=nf*2, \
                        kernel_size=5, stride=1, padding=2, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf*2, out_nc=nf*2, \
                        kernel_size=5, stride=1, padding=2, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            )

        self.dec_l2_out = B.conv_block(in_nc=nf*2, out_nc=3, \
                            kernel_size=1, stride=1, padding=0, dilation=1, \
                            bias=True, groups=1, norm_type=None, act_type=None)

        self.dec_up_l21 = B.tconv_block(in_nc=nf*2, out_nc=nf, \
                            kernel_size=4, stride=2, padding=1, dilation=1, \
                            bias=True, groups=1, norm_type=None, act_type=act_type)

        self.dec_l1_out = nn.Sequential(
            B.conv_block(in_nc=nf, out_nc=nf, \
                        kernel_size=5, stride=1, padding=2, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf, out_nc=nf, \
                        kernel_size=5, stride=1, padding=2, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf, out_nc=nf, \
                        kernel_size=5, stride=1, padding=2, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf, out_nc=nf, \
                        kernel_size=5, stride=1, padding=2, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=act_type),
            B.conv_block(in_nc=nf, out_nc=3, \
                        kernel_size=1, stride=1, padding=0, dilation=1, \
                        bias=True, groups=1, norm_type=None, act_type=None)
            )


    def encode(self, x, map):

        e1 = self.enc0(x)
        e1 = self.enc1_1(e1, map)
        e1 = self.enc1_2(e1, map)
        e1 = self.enc1_3(e1, map)
        res1 = e1

        e2 = self.enc2_0(e1)
        e2 = self.enc2_1(e2, map)
        e2 = self.enc2_2(e2, map)
        e2 = self.enc2_3(e2, map)
        res2 = e2

        e3 = self.enc3_0(e2)
        e3 = self.enc3_1(e3, map)
        e3 = self.enc3_2(e3, map)
        feat = self.enc3_3(e3, map)

        return feat, res1, res2

    def decode_l3(self, x, map):

        x = self.dec_l3_0(x, map)
        x = self.dec_l3_1(x, map)
        x = self.dec_l3_2(x, map)
        x = self.dec_l3_3(x)

        return x

    def decode_l21(self, x, res1, res2):

        # import ipdb; ipdb.set_trace()
        x = self.dec_0(x)

        x = self.dec_1(torch.add(x, res2))

        l2 = self.dec_l2_out(x)

        x = self.dec_up_l21(x)

        l1 = self.dec_l1_out(torch.add(x, res1))

        return l2, l1

    def forward(self, x, map):

        feat, res1, res2 = self.encode(x, map)

        l3_n = self.decode_l3(feat, map)
        l3_n = torch.sigmoid(l3_n)
        l2_n, l1_n = self.decode_l21(feat, res1, res2)
        l2_n = torch.tanh(l2_n)
        l1_n = torch.tanh(l1_n)

        out = recomposition([l3_n, l2_n, l1_n])

        return out, l1_n, l2_n, l3_n

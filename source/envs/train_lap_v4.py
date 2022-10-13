import os
import time

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
import utils.colors as cl
import utils.metrics as mcs
import utils.train_utils as tu
from models.LAPNN import LAP_v4, LAP_v4_1, gauss_kernel, laplacian_pyramid
from utils.metrics import calc_psnr, calc_ssim
from utils.perceptual import PerceptualLoss
from utils.print_utils import printProgressBar
from utils.weight_initializers import init_weights


class trainEnv(object):

    def __init__(self, opt, dataset_train, dataset_val):

        self.device = opt.device
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val

        self.save_dir = opt.save_dir
        self.check_path = opt.check_path

        print("\n===> Building models")

        self.model = LAP_v4_1(nf=64, device=self.device)

        self.MSELoss = nn.L1Loss()

        self.optim = optim.Adam(self.model.parameters(),
                                lr=opt.lr,
                                betas=(opt.beta, 0.999),
                                weight_decay=1e-8
                                )

        self.gauss_kernel = gauss_kernel(channels=3, device=self.device)

        # Checkpoint loading

        if opt.chkp != '':
            if opt.chkpe == -1:
                self.load_checkpoint(opt.chkp, 'last', test=False)
            else:
                self.load_checkpoint(opt.chkp, str(opt.chkpe), test=False)

    def train(self, epoch, log_str):

        log_str.clear()
        self.model.eval()
        print('\n')
        lr = self.optim.param_groups[0]['lr']
        print('\n \033[92m|NEW LR:\033[0m ' + str(lr))
        loss_mean = 0
        lsd = 0

        for i, batch in enumerate(self.dataset_train):

            st = time.time()
            # DATA
            inputt, target = batch[0], batch[1]

            if self.cuda:
                inputt = inputt.to(self.device)
                target = target.to(self.device)

            load_t = time.time() - st
            st = time.time()

            self.model.zero_grad()

            out, l1, l2, l3 = self.model(inputt)

            # Loss
            # t_l1, t_l2, t_l3 = laplacian_pyramid(target, self.gauss_kernel, max_levels=3)
            #
            # # LOW
            # L_l3 = self.MSELoss(l3, t_l3)
            #
            # # MID
            # _,_,hh,ww = l2.shape
            # up = F.interpolate(l3, size=(hh, ww), mode='bicubic', align_corners=False)
            # l2 = up.detach() + l2
            #
            # up = F.interpolate(t_l3, size=(hh, ww), mode='bicubic', align_corners=False)
            # t_l2 = up + t_l2
            #
            # L_l2 = self.MSELoss(l2, t_l2)
            #
            # # HIGH
            # _,_,hh,ww = l1.shape
            # up = F.interpolate(l2, size=(hh, ww), mode='bicubic', align_corners=False)
            # l1 = up.detach() + l1
            #
            # up = F.interpolate(t_l2, size=(hh, ww), mode='bicubic', align_corners=False)
            # t_l1 = up + t_l1
            #
            # L_l1 = self.MSELoss(l1, t_l1)

            # COMPLETE
            L_tot = self.MSELoss(out, target)

            # Old
            # Loss = 4*L_l3 + 2*L_l2 + L_l1

            # lev1: 2, lev2: 3, lev3: 1. SSIM
            # Loss = L_l3 + 3*L_l2 + 2*L_l1

            # lev1: 3, lev2: 4, lev3: 1. PSNR
            # Loss = L_l3 + 4*L_l2 + 3*L_l1

            # all same
            # Loss = L_l3 + L_l2 + L_l1
            Loss = L_tot

            Loss.backward()

            self.optim.step()

            loss_mean += Loss.item()

            train_t = time.time() - st

            # Print and images save ------------------------------------------

            # if i % 50 == 0:
            #     from utils.grad import plot_grad_flow; plot_grad_flow(self.model.named_parameters())
            #     import ipdb; ipdb.set_trace()

            if i % 50 == 0:
                print('[{}][{}/{}]\t\033[92mLoss l1:{:.8f}  l2:{:.8f}  l3:{:.8f} \n\t\033[94mLoad time: {:.4} \tBatch time: {:.4} \033[0m'.format(
                    epoch, i, len(self.dataset_train),
                    L_tot.item(), 0, 0, load_t, train_t))
                # L_l1.item(), L_l2.item(), L_l3.item(), load_t, train_t))

            # if i % 200 == 0:
            #     vutils.save_image(target, self.save_dir + '/gen_images/train/'+ format(epoch, '03d') + '_' + format(i, '03d') + '_real_sample.png')
            #     vutils.save_image(inputt, self.save_dir + '/gen_images/train/'+ format(epoch, '03d') + '_' + format(i, '03d') + '_input_sample.png')
            #     vutils.save_image(out, self.save_dir + '/gen_images/train/'+ format(epoch, '03d') + '_' + format(i, '03d') + '_out_sample.png')

        loss_mean = loss_mean / len(self.dataset_train)
        lsd = lsd / len(self.dataset_train)
        log_str.append(epoch)
        log_str.append(loss_mean)
        log_str.append(0)

    def validate(self, epoch, log_str):

        with torch.no_grad():

            print("\n\n==> VALIDATION")

            self.model.eval()
            err_mean = 0
            psnr_avg = 0
            ssim_avg = 0

            dir = 'epoch_' + format(epoch, '03d')

            if not os.path.exists(self.save_dir + '/gen_images/validation/' + dir):
                os.makedirs(self.save_dir + '/gen_images/validation/' + dir)

            printProgressBar(0, len(self.dataset_train),
                             prefix='Validation:', suffix='', length=50)

            for i, batch in enumerate(self.dataset_val, 1):

                inputt, target = batch[0], batch[1]

                if self.cuda:
                    inputt = inputt.to(self.device)
                    target = target.to(self.device)

                _, _, h, w = inputt.shape

                out, _, _, _ = self.model(inputt)

                err = self.MSELoss(out, target)

                err_mean += err.item()

                out = torch.clamp(out, 0, 1)

                # Output image saving

                res_a = out[0].cpu().numpy().transpose(1, 2, 0) * 255
                inpt = inputt[0].cpu().numpy().transpose(1, 2, 0) * 255
                tar_a = target[0].cpu().numpy().transpose(1, 2, 0) * 255

                # import ipdb; ipdb.set_trace()
                res_a = res_a.astype(np.uint8)
                inpt = inpt.astype(np.uint8)
                tar_a = tar_a.astype(np.uint8)

                psnr = mcs.calc_psnr(res_a, tar_a)
                ssim = mcs.calc_ssim(res_a, tar_a)

                psnr_avg += psnr.mean()
                ssim_avg += ssim.mean()

                vutils.save_image(inputt, self.save_dir + '/gen_images/validation/' + dir +
                                  '/val_' + format(epoch, '03d') + '_' + format(i, '03d') + '_input.png')
                vutils.save_image(target, self.save_dir + '/gen_images/validation/' + dir +
                                  '/val_' + format(epoch, '03d') + '_' + format(i, '03d') + '_target.png')
                vutils.save_image(out, self.save_dir + '/gen_images/validation/' + dir +
                                  '/val_' + format(epoch, '03d') + '_' + format(i, '03d') + '_enhanced.png')

                printProgressBar(i + 1, len(self.dataset_val),
                                 prefix='Validation:', suffix='', length=50)

            err_mean = err_mean / len(self.dataset_val)
            psnr_avg = psnr_avg / len(self.dataset_val)
            ssim_avg = ssim_avg / len(self.dataset_val)

            print("Validation mean error: {} PSNR: {} SSIM: {}".format(
                err_mean, psnr_avg, ssim_avg
            ))
            log_str.append(err_mean)
            log_str.append(psnr_avg)
            log_str.append(ssim_avg)

    def cuda(self):
        self.model.to(self.device)
        self.MSELoss.to(self.device)

    def adjust_lr(self, sf=0.1):

        tu.adjust_learning_rate(self.optim, sf=sf)

    def load_checkpoint(self, path, epoch, test=False):

        name = type(self.model).__name__

        if epoch == 'last':
            path = path + "/last_ckp/"
        else:
            path = path + "/checkpoints/"

        tu.load_model(self.model, path + '/net' + name +
                      '_epoch' + str(epoch) + '.pth')

        if not test:
            tu.load_optimizer(self.optim, path + '/optim' +
                              name + '_epoch' + str(epoch) + '.pth', self.device)

    def save_checkpoint(self, path, epoch):

        name = type(self.model).__name__
        # Checkpoint save
        tu.save_model(self.model.state_dict(), path, epoch, name)
        tu.save_model(self.model.state_dict(), path +
                      '/../last_ckp', 'last', name)

        tu.save_optim(self.optim.state_dict(), path, epoch, name + '_opt')
        tu.save_optim(self.optim.state_dict(), path +
                      '/../last_ckp', 'last', name)

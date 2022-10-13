import argparse

import torch


class OptionParser(object):

    def __init__(self):

        self.opt = self.parse_opt()

        if self.opt.lrdwn is None:
            self.opt.lrdwn = []

        self.opt.check_path = self.opt.save_dir + '/checkpoints'

        # GPU board availability check
        self.opt.cuda = False
        if torch.cuda.is_available() and self.opt.device != 'cpu':
            self.opt.cuda = True

    def get_opts(self):
        return self.opt

    def parse_opt(self):

        parser = argparse.ArgumentParser(
            description='CR-GAN implementation in PyTorch')

        parser.add_argument("--epochs", type=int, default=2000,
                            help="number of epochs to train for")
        parser.add_argument('--batch_size', type=int,
                            default=16, help='training batch size')
        parser.add_argument("--device", type=str, default='cuda:0',
                            help="Device for training [cpu, cuda:0, cuda:1]")

        parser.add_argument("--augm", action="store_true",
                            default=False, help="Data augmentation flag")
        parser.add_argument("--lr", type=float, default=2e-4,
                            help="Learning Rate for Generator.")
        parser.add_argument('--beta', type=float, default=0.9,
                            help='beta1 for adam (Generator).')

        parser.add_argument(
            '--lrdwn', nargs='*', help='Learning rate decreasing epochs', required=False)

        parser.add_argument("--nf", type=int, default=8,
                            help="Number of channel for Encoder.")

        parser.add_argument('--images_dir', type=str, help='dataset directory')
        parser.add_argument("--save_dir", type=str, default='./',
                            help="directory where to save output data and checkpoints")

        parser.add_argument('--threads', type=int, default=4,
                            help='number of threads for data loader to use')

        parser.add_argument('--chkp', type=str, default='',
                            help='checkpoint path to save')
        parser.add_argument('--chkpe', type=int, default=-1,
                            help='epoch to load [-1: last epoch]')

        parser.add_argument('--log', type=str, default='',
                            help='Log file to continue')

        opt = parser.parse_args()

        return opt

    def print_options(self):

        print('\n===> Options \
                \n \
                \nNumber of epochs: {} \
                \nBatch size: {} \
                \nLearning Rate: {} \
                \nDecreasing lr at: {} \
                \nGPU: {} \
                \nData augmentation: {} \
                \nSave directory: {}'.format(
            self.opt.epochs, self.opt.batch_size, self.opt.lr, self.opt.lrdwn, self.opt.device, self.opt.augm, self.opt.save_dir
        ))

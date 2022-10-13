import csv
import os

import numpy as np

import torch
from data.dataloader import load_dataset
# from envs.train_map_est import trainEnv
# from envs.train_lap_v4_map import trainEnv
from envs.train_lap_v4 import trainEnv
from option_parser import OptionParser
from utils.compare import compute_difference_cl
from utils.image_functions import transform_image
from utils.print_utils import printProgressBar

torch.backends.cudnn.deterministic=True

################################################################################

if __name__ == '__main__':

    ###################### Option Parsing ######################################
    OP = OptionParser()
    OP.print_options()
    opt = OP.get_opts()

    ### Setting cuda
    cuda_en = False
    if opt.device != 'cpu':
        cuda_en = True

    ### Seed setting for random operations
    torch.manual_seed(123123)
    if cuda_en:
        torch.cuda.manual_seed_all(123123)
    np.random.seed(123123)

    ###################### Save Dir Check ######################################

    dirs = []
    dirs.append(opt.save_dir)
    dirs.append(opt.save_dir + '/gen_images/')
    dirs.append(opt.save_dir + '/gen_images/validation')
    dirs.append(opt.save_dir + '/gen_images/train')
    dirs.append(opt.check_path)
    dirs.append(opt.save_dir+'/last_ckp')

    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

    ###################### Log File ############################################

    if opt.chkp == '':
        log_file = open(opt.save_dir + '/log.csv', "w")
    else:
        lg = opt.chkp + '/log.csv'
        log_file = open(lg, "a")
    wr = csv.writer(log_file, quoting=csv.QUOTE_ALL)

    ##################### Data Loader initialization ###########################

    training_data_loader, validation_data_loader = load_dataset(opt)

    ############################ Training Env ##################################
    tenv = trainEnv(opt, training_data_loader, validation_data_loader)

    print("\n===> Setting GPU")
    if cuda_en:
        tenv.cuda()

    ############################### TRAIN ##########################################
    print("\n===> Training")

    log_str = []

    ### Main

    for epoch in range(0, opt.epochs):

        if str(epoch) in opt.lrdwn:
            tenv.adjust_lr(sf=0.1)

        tenv.train(epoch, log_str)

        tenv.validate(epoch, log_str)

        wr.writerow(log_str)
        log_file.flush()
        log_str = []

        tenv.save_checkpoint(opt.check_path, epoch)
        print('\nEpoch {} finished!'.format(epoch))


    log_file.close()

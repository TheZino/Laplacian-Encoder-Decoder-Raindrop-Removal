
import argparse
import csv
import os
import random
import time

import cv2
import numpy as np
import skimage.io as io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
from models.LAPNN import LAP_v4_1, laplacian_pyramid, recomposition
from models.Unet import Unet3
from torchvision.transforms import Compose, Normalize, ToTensor
from utils.image_padder import PadStride
from utils.metrics import calc_psnr, calc_ssim


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--model_map", type=str)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--gt_dir", type=str)
    parser.add_argument("--epoch", type=str, default="last")
    parser.add_argument("--device", type=str, default="cpu")
    opt = parser.parse_args()
    return opt

def align_to_four(img):
    # print ('before alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    #align to four
    a_row = int(img.shape[0]/4)*4
    a_col = int(img.shape[1]/4)*4
    img = img[0:a_row, 0:a_col]
    # print ('after alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    return img

def to_out_numpy(out, size):

    h,w = size

    out = out[-1]

    _,hh,ww = out.shape

    nh = hh - h
    nw = ww - w

    out = out[:, int(nh/2) : hh - int(nh/2),  int(nw/2) : ww - int(nw/2) ]
    # out = (out + 1)/2
    out = torch.clamp(out, 0, 1)

    out = out.cpu().data
    out = out.numpy()
    out = out.transpose((1,2,0))
    out = out*255.

    return out




def predict(image, gt, opt, im_name):
    # image = np.array(image, dtype='float32')/255.
    # image = image.transpose((2, 0, 1))
    # image = image[np.newaxis, :, :, :]
    # image = torch.from_numpy(image)
    # image = image.to(device)

    device = opt.device
    path = opt.output_dir + '/images/'

    h,w,_ = image.shape

    to_tensor = Compose([
            PadStride(32, fill=-1),
            ToTensor()
            ])

    image = to_tensor(image).unsqueeze(0)
    image = image.to(device)

    gt_t = to_tensor(gt).unsqueeze(0)
    gt_t = gt_t.to(device)

    with torch.no_grad():
        out = model(image)

    [l1gt, l2gt, l3gt] = laplacian_pyramid(gt_t, model.gauss_kernel, max_levels=3)
    l3o = out[3]
    l2o = out[2]
    l1o = out[1]

    ### L3 substitution
    tl3 = recomposition([l3gt, l2o, l1o])

    ### L2 substitution
    tl2 = recomposition([l3o, l2gt, l1o])

    ### L1 substitution
    tl1 = recomposition([l3o, l2o, l1gt])


    ### Output
    out = out[0]
    out = to_out_numpy(out, [h,w])
    tl3 = to_out_numpy(tl3, [h,w])
    tl2 = to_out_numpy(tl2, [h,w])
    tl1 = to_out_numpy(tl1, [h,w])

    l1o = to_out_numpy(l1o, [h,w])
    l2o = to_out_numpy(l2o, [h/2,w/2])
    l3o = to_out_numpy(l3o, [h/4,w/4])

    l1gt = to_out_numpy(l1gt, [h,w])
    l2gt = to_out_numpy(l2gt, [h/2,w/2])
    l3gt = to_out_numpy(l3gt, [h/4,w/4])

    im_name = im_name.split('.')[0]

    # import ipdb; ipdb.set_trace()


    io.imsave(path + '/' + im_name + '_out.png', np.array(out, dtype = 'uint8'))
    io.imsave(path + '/' + im_name + '_tl1.png', np.array(tl1, dtype = 'uint8'))
    io.imsave(path + '/' + im_name + '_tl2.png', np.array(tl2, dtype = 'uint8'))
    io.imsave(path + '/' + im_name + '_tl3.png', np.array(tl3, dtype = 'uint8'))
    io.imsave(path + '/' + im_name + '_l1o.png', np.array(l1o, dtype = 'uint8'))
    io.imsave(path + '/' + im_name + '_l2o.png', np.array(l2o, dtype = 'uint8'))
    io.imsave(path + '/' + im_name + '_l3o.png', np.array(l3o, dtype = 'uint8'))
    io.imsave(path + '/' + im_name + '_l1gt.png', np.array(l1gt, dtype = 'uint8'))
    io.imsave(path + '/' + im_name + '_l2gt.png', np.array(l2gt, dtype = 'uint8'))
    io.imsave(path + '/' + im_name + '_l3gt.png', np.array(l3gt, dtype = 'uint8'))


    return out, tl3, tl2, tl1


if __name__ == '__main__':
    opt = get_opt()
    results = []
    model = LAP_v4_1(nf = 64, device = opt.device)
    model.to(opt.device)
    model_map = Unet3(out_ch = 1, nf = 16)
    model_map.to(opt.device)
    if opt.epoch != 'last':
        model_w = 'checkpoints/net' + model.__class__.__name__ + '_epoch' + opt.epoch + '.pth'
    else:
        model_w = 'checkpoints/net' + model.__class__.__name__ + '_epochlast.pth'
    model.load_state_dict(torch.load(opt.model + model_w, map_location='cpu'))
    model_map.load_state_dict(torch.load(opt.model_map, map_location='cpu'))


    input_list = sorted(os.listdir(opt.input_dir))
    gt_list = sorted(os.listdir(opt.gt_dir))
    num = len(input_list)

    cumulative_psnr = 0
    cumulative_ssim = 0
    cumulative_psnr1 = 0
    cumulative_ssim1 = 0
    cumulative_psnr2 = 0
    cumulative_ssim2 = 0
    cumulative_psnr3 = 0
    cumulative_ssim3 = 0

    exp_id = opt.model.split('/')[-2]


    opt.output_dir = opt.output_dir + '/' + exp_id + '/'

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    if not os.path.exists(opt.output_dir + '/images/'):
        os.makedirs(opt.output_dir + '/images/')

    log_file = open(opt.output_dir + '/results_epoch' + opt.epoch + '.csv', "w")
    wr = csv.writer(log_file, quoting=csv.QUOTE_ALL)

    for i in range(num):
        print ('Processing image: %s'%(input_list[i]))

        ## Reading data and conversion from BGR to RGB
        img = cv2.imread(opt.input_dir + input_list[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(opt.gt_dir + gt_list[i])
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

        ## Padding like operation to fir the division in Unet
        img = align_to_four(img)
        gt = align_to_four(gt)

        result, tl3, tl2, tl1 = predict(img, gt, opt, input_list[i])


        # print("max: {}\tmin:{}".format(result.max(), result.min()))
        result = np.array(result, dtype = 'uint8')
        tl3 = np.array(tl3, dtype = 'uint8')
        tl2 = np.array(tl2, dtype = 'uint8')
        tl1 = np.array(tl1, dtype = 'uint8')

        cur_psnr = calc_psnr(result, gt)
        cur_ssim = calc_ssim(result, gt)
        print('PSNR is %.4f and SSIM is %.4f'%(cur_psnr, cur_ssim))
        cumulative_psnr += cur_psnr
        cumulative_ssim += cur_ssim

        cur_psnr3 = calc_psnr(tl3, gt)
        cur_ssim3 = calc_ssim(tl3, gt)
        print('PSNR is %.4f and SSIM is %.4f'%(cur_psnr3, cur_ssim3))
        cumulative_psnr3 += cur_psnr3
        cumulative_ssim3 += cur_ssim3

        cur_psnr2 = calc_psnr(tl2, gt)
        cur_ssim2 = calc_ssim(tl2, gt)
        print('PSNR is %.4f and SSIM is %.4f'%(cur_psnr2, cur_ssim2))
        cumulative_psnr2 += cur_psnr2
        cumulative_ssim2 += cur_ssim2

        cur_psnr1 = calc_psnr(tl1, gt)
        cur_ssim1 = calc_ssim(tl1, gt)
        print('PSNR is %.4f and SSIM is %.4f'%(cur_psnr1, cur_ssim1))
        cumulative_psnr1 += cur_psnr1
        cumulative_ssim1 += cur_ssim1

        results = [input_list[i], cur_psnr, cur_ssim, cur_psnr3, cur_ssim3, cur_psnr2, cur_ssim2, cur_psnr1, cur_ssim1]
        wr.writerow(results)
        log_file.flush()
        results = []

    print('In testing dataset, PSNR is %.4f and SSIM is %.4f'%(cumulative_psnr/num, cumulative_ssim/num))
    print('In testing dataset, PSNR is %.4f and SSIM is %.4f'%(cumulative_psnr3/num, cumulative_ssim3/num))
    print('In testing dataset, PSNR is %.4f and SSIM is %.4f'%(cumulative_psnr2/num, cumulative_ssim2/num))
    print('In testing dataset, PSNR is %.4f and SSIM is %.4f'%(cumulative_psnr1/num, cumulative_ssim1/num))
    results = ['AVG', cumulative_psnr/num, cumulative_ssim/num, cumulative_psnr3/num, cumulative_ssim3/num, cumulative_psnr2/num, cumulative_ssim2/num, cumulative_psnr1/num, cumulative_ssim1/num]
    wr.writerow(results)
    log_file.flush()
    results = []

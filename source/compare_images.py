import argparse
import csv
import os

import cv2
import numpy as np

from utils.metrics import calc_psnr, calc_ssim


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--gt_dir", type=str)
    opt = parser.parse_args()
    return opt


def align_to_four(img):
    # print ('before alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    # align to four
    a_row = int(img.shape[0] / 4) * 4
    a_col = int(img.shape[1] / 4) * 4
    img = img[0:a_row, 0:a_col]
    # print ('after alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    return img


if __name__ == '__main__':
    opt = get_opt()
    results = []

    input_list = sorted(os.listdir(opt.input_dir))
    gt_list = sorted(os.listdir(opt.gt_dir))
    num = len(input_list)
    cumulative_psnr = 0
    cumulative_ssim = 0

    opt.output_dir = opt.output_dir + '/'

    for dir in [opt.output_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    log_file = open(opt.output_dir + '/results.csv', "w")
    wr = csv.writer(log_file, quoting=csv.QUOTE_ALL)

    for i in range(num):
        print('Processing image: {}\tgt: {}'.format(input_list[i], gt_list[i]))

        # Reading data and conversion from BGR to RGB
        img = cv2.imread(opt.input_dir + '/' + input_list[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(opt.gt_dir + '/' + gt_list[i])
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

        # fix shifted images
        img = align_to_four(img)
        gt = align_to_four(gt)

        result = np.array(img, dtype='uint8')

        assert (
            result.shape == gt.shape
        ), "dimensions must agree [input: {}, gt:{}]".format(result.shape,
                                                             gt.shape)

        cur_psnr = calc_psnr(result, gt)
        cur_ssim = calc_ssim(result, gt)
        print('PSNR is %.4f and SSIM is %.4f' % (cur_psnr, cur_ssim))
        cumulative_psnr += cur_psnr
        cumulative_ssim += cur_ssim

        results = [input_list[i], cur_psnr, cur_ssim]
        wr.writerow(results)
        log_file.flush()
        results = []

        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    print(
        'In testing dataset, PSNR is %.4f and SSIM is %.4f'
        % (cumulative_psnr / num, cumulative_ssim / num)
    )
    results = ['AVG', cumulative_psnr / num, cumulative_ssim / num]
    wr.writerow(results)
    log_file.flush()
    results = []

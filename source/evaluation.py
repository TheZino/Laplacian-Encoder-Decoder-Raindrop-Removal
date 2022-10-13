import argparse
import csv
import glob

import cv2
import numpy as np
import utils.metrics as mcs

parser = argparse.ArgumentParser(description='PSNR - SSIM Evaluation')

parser.add_argument("--dir_enh", '-de', type=str, default='./outputs',
                    help="directory where to find enhanced images")
parser.add_argument("--dir_gt", '-dg', type=str, default='./gt',
                    help="directory where to find gt images")
parser.add_argument("--out_dir", '-o', type=str,
                    default='./', help="output file")

opt = parser.parse_args()


en_files = [f for f in glob.glob(opt.dir_enh + "/*.png")]
en_files.sort()
gt_files = [f for f in glob.glob(opt.dir_gt + "/*.png")]
gt_files.sort()


def align_to_four(img):
    # print ('before alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    # align to four
    a_row = int(img.shape[0] / 4) * 4
    a_col = int(img.shape[1] / 4) * 4
    img = img[0:a_row, 0:a_col]
    # print ('after alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    return img


psnr_avg = 0
ssim_avg = 0

print("== Evaluating {} images".format(len(en_files)))

with open(opt.out_dir + 'results.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile)

    for en_file, gt_file in zip(en_files, gt_files):

        name = en_file.split('/')[-1]

        im_en = cv2.imread(en_file)
        im_gt = cv2.imread(gt_file)

        im_en = align_to_four(im_en)
        im_gt = align_to_four(im_gt)

        im_en = np.array(im_en, dtype='uint8')
        im_gt = np.array(im_gt, dtype='uint8')

        # RANGE MUST BE [0, 255]
        psnr = mcs.calc_psnr_cv(im_en, im_gt)
        psnr_avg += psnr
        ssim = mcs.calc_ssim_cv(im_en, im_gt)
        ssim_avg += ssim

        spamwriter.writerow([name, psnr, ssim])

    psnr_avg /= len(en_files)
    ssim_avg /= len(en_files)
    spamwriter.writerow(['AVG', psnr_avg, ssim_avg])

print("== DONE! \n\tAVG PSNR: {} \tAVG SSIM: {}\n\t saved in {}".format(
    psnr_avg, ssim_avg, opt.out_dir + 'results.csv'))

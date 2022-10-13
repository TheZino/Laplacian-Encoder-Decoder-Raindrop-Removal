import argparse
import csv
import os
import time

import cv2
import numpy as np
import torch
import torchvision.transforms as trs
from models.LAPNN import LAP_v4_1
from utils.image_padder import PadStride
from utils.metrics import calc_psnr, calc_ssim


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--epoch", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    opt = parser.parse_args()
    return opt


def predict(image, device):
    h, w, _ = image.shape

    to_tensor = trs.Compose([
        PadStride(32, fill=-1),
        trs.ToTensor()
    ])

    image = to_tensor(image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        out = model(image)
        out = out[0]

    out = out[-1]

    _, hh, ww = out.shape

    nh = hh - h
    nw = ww - w

    out = out[:, int(nh / 2): hh - int(nh / 2), int(nw / 2): ww - int(nw / 2)]
    out = torch.clamp(out, 0, 1)

    out = out.cpu().data
    out = out.numpy()
    out = out.transpose((1, 2, 0))
    out = out * 255.

    return out


if __name__ == '__main__':
    opt = get_opt()
    results = []
    model = LAP_v4_1(nf=64, device=opt.device)
    model.to(opt.device)

    model.load_state_dict(torch.load(opt.model, map_location='cpu'))

    input_list = sorted([f for f in os.listdir(opt.input_dir)
                         if os.path.isfile(os.path.join(opt.input_dir, f))])

    num = len(input_list)

    exp_id = opt.model.split('/')[-2]

    im_output_dir = opt.output_dir

    for dir in [opt.output_dir, im_output_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    tt = 0
    for i in range(num):
        print('Processing image: %s' % (input_list[i]))

        # Reading data and conversion from BGR to RGB
        img = cv2.imread(opt.input_dir + input_list[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        start = time.time()
        result = predict(img, opt.device)
        tt += time.time() - start

        result = np.array(result, dtype='uint8')

        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(im_output_dir + input_list[i], result)

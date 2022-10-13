import skimage.io as io
import torch
from torchvision.transforms import ToTensor
from torchvision.utils import save_image

from LAPNN import gauss_kernel, laplacian_pyramid, recomposition

tt = ToTensor()

im = io.imread('/fastsets/raindrop/test/data/0_rain.png')
im_t = tt(im)

im_t = im_t.unsqueeze(0)

im_t = torch.cat([im_t, im_t], 0)

kernel = gauss_kernel()

asd = laplacian_pyramid(im_t, kernel, max_levels=4)

im_r = recomposition(asd)

i =0
for ten in asd:
    save_image(asd[i], 'l'+str(i+1)+'.png')
    i+=1

save_image(im_r, 'recon.png')

import ipdb; ipdb.set_trace()

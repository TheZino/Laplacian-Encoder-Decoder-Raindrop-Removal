import numpy as np
import skimage.color as color
from skimage.metrics import structural_similarity as compare_ssim


def compute_difference(im, imgt, SRF = 4):
    '''
    COMPUTE_DIFFERENCE: compute image quality
    Input:
        - im:   super-resolved image (numpy array HxWxC) [0,1]
        - imgt: groundtruth high-resolution image (numpy array HxWxC) [0,1]
        - SRF:  super-resolution factor
    Output:
        - psnr: Peak signal-to-noise ratio
        - ssim: Structural similarity index
    '''
    # =========================================================================
    # Retrieve only luminance channel
    # =========================================================================
    h,w,c = im.shape
    h_gt,w_gt,c_gt = im.shape

    im[im>1] = 1
    im[im<0] = 0

    im = color.rgb2ycbcr(im)
    imgt = color.rgb2ycbcr(imgt)
    im_y = im[:,:,0]
    imgt_y = imgt[:,:,0]

    # =========================================================================
    # Remove border pixels as some methods (e.g., A+) do not predict border pixels
    # =========================================================================
    cropPix = SRF
    im_y = np.around(im_y[cropPix:h-cropPix, cropPix:w-cropPix]).astype(np.double)
    imgt_y = np.around(imgt_y[cropPix:h_gt-cropPix, cropPix:w_gt-cropPix]).astype(np.double)

    # =========================================================================
    # Compute Peak signal-to-noise ratio (PSNR)
    # =========================================================================
    mse = np.mean(np.mean((im_y-imgt_y)**2,1),0)
    psnr = 10*np.log10(255*255/mse)

    # =========================================================================
    # Compute Structural similarity index (SSIM index)
    # =========================================================================
    ssim = compare_ssim(im_y, imgt_y, K1=0.01, K2=0.03, data_range=256, gaussian_weights=True)

    return psnr, ssim


def compute_difference_cl(im, imgt, SRF = 4):
    '''
    COMPUTE_DIFFERENCE: compute image quality
    Input:
        - im:   super-resolved image (numpy array HxWxC) [0,255]
        - imgt: groundtruth high-resolution image (numpy array HxWxC) [0,255]
        - SRF:  super-resolution factor
    Output:
        - psnr: Peak signal-to-noise ratio
        - ssim: Structural similarity index
    '''
    # =========================================================================
    # Retrieve only luminance channel
    # =========================================================================
    h,w,c = im.shape
    h_gt,w_gt,c_gt = im.shape

    im[im>1] = 1
    im[im<0] = 0

    # =========================================================================
    # Remove border pixels as some methods (e.g., A+) do not predict border pixels
    # =========================================================================
    cropPix = SRF
    im =im[cropPix:h-cropPix, cropPix:w-cropPix].astype(np.double)
    imgt = imgt[cropPix:h_gt-cropPix, cropPix:w_gt-cropPix].astype(np.double)

    # =========================================================================
    # Compute Peak signal-to-noise ratio (PSNR)
    # =========================================================================
    mse = np.mean(np.mean((im-imgt)**2,1),0)
    psnr = 10*np.log10(255*255/mse)

    # =========================================================================
    # Compute Structural similarity index (SSIM index)
    # =========================================================================
    ssim = compare_ssim(im, imgt, K1=0.01, K2=0.03, data_range=256, gaussian_weights=True, multichannel=True)

    return psnr, ssim

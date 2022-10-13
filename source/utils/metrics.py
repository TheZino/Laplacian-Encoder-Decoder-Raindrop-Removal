import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calc_psnr(im1, im2):
    '''
    im1 and im2 range [0,255]
    '''
    # im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    # im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im1_y = cv2.cvtColor(im1, cv2.COLOR_RGB2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_RGB2YCR_CB)[:, :, 0]
    return peak_signal_noise_ratio(im1_y, im2_y)

def calc_ssim(im1, im2):
    '''
    im1 and im2 range [0,255]
    '''
    # im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    # im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im1_y = cv2.cvtColor(im1, cv2.COLOR_RGB2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_RGB2YCR_CB)[:, :, 0]
    return structural_similarity(im1_y, im2_y)



def calc_psnr_cv(im1, im2):
    '''
    im1 and im2 range [0,255]
    '''
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    # im1_y = cv2.cvtColor(im1, cv2.COLOR_RGB2YCR_CB)[:, :, 0]
    # im2_y = cv2.cvtColor(im2, cv2.COLOR_RGB2YCR_CB)[:, :, 0]
    return peak_signal_noise_ratio(im1_y, im2_y)

def calc_ssim_cv(im1, im2):
    '''
    im1 and im2 range [0,255]
    '''
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    # im1_y = cv2.cvtColor(im1, cv2.COLOR_RGB2YCR_CB)[:, :, 0]
    # im2_y = cv2.cvtColor(im2, cv2.COLOR_RGB2YCR_CB)[:, :, 0]
    return structural_similarity(im1_y, im2_y)

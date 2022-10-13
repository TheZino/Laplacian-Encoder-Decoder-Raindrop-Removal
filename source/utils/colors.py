from warnings import warn

import numpy as np
from scipy import linalg

import torch


# Helper for the creation of module-global constant tensors
def _t(data):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # TODO inherit this
    device = torch.device("cpu") # TODO inherit this
    return torch.tensor(data, requires_grad=False, dtype=torch.float32, device=device)

# Helper for color matrix multiplication
def _mul(coeffs, image):
    # This is implementation is clearly suboptimal.  The function will
    # be implemented with 'einsum' when a bug in pytorch 0.4.0 will be
    # fixed (Einsum modifies variables in-place #7763).
    coeffs = coeffs.to(image.device)
    r0 = image[:, 0:1, :, :].repeat(1, 3, 1, 1) * coeffs[:, 0].view(1, 3, 1, 1)
    r1 = image[:, 1:2, :, :].repeat(1, 3, 1, 1) * coeffs[:, 1].view(1, 3, 1, 1)
    r2 = image[:, 2:3, :, :].repeat(1, 3, 1, 1) * coeffs[:, 2].view(1, 3, 1, 1)
    return r0 + r1 + r2
    # return torch.einsum("dc,bcij->bdij", (coeffs.to(image.device), image))


_RGB_TO_YCBCR = _t([[0.257, 0.504, 0.098], [-0.148, -0.291, 0.439], [0.439 , -0.368, -0.071]])
_YCBCR_OFF = _t([0.063, 0.502, 0.502]).view(1, 3, 1, 1)


def rgb2ycbcr(rgb):
    """sRGB to YCbCr conversion."""
    clip_rgb=False
    if clip_rgb:
        rgb = torch.clamp(rgb, 0, 1)
    return _mul(_RGB_TO_YCBCR, rgb) + _YCBCR_OFF.to(rgb.device)


def ycbcr2rgb(rgb):
    """YCbCr to sRGB conversion."""
    clip_rgb=False
    rgb = _mul(torch.inverse(_RGB_TO_YCBCR), rgb - _YCBCR_OFF.to(rgb.device))
    if clip_rgb:
        rgb = torch.clamp(rgb, 0, 1)
    return rgb

##### LAB RGB


xyz_from_rgb = torch.Tensor([[0.412453, 0.357580, 0.180423],
                         [0.212671, 0.715160, 0.072169],
                         [0.019334, 0.119193, 0.950227]])

rgb_from_xyz = torch.Tensor(linalg.inv(xyz_from_rgb))

illuminants = \
    {"A": {'2': (1.098466069456375, 1, 0.3558228003436005),
           '10': (1.111420406956693, 1, 0.3519978321919493)},
     "D50": {'2': (0.9642119944211994, 1, 0.8251882845188288),
             '10': (0.9672062750333777, 1, 0.8142801513128616)},
     "D55": {'2': (0.956797052643698, 1, 0.9214805860173273),
             '10': (0.9579665682254781, 1, 0.9092525159847462)},
     "D65": {'2': torch.Tensor([(0.95047, 1., 1.08883)]),   # This was: `lab_ref_white`
             '10': (0.94809667673716, 1, 1.0730513595166162)},
     "D75": {'2': (0.9497220898840717, 1, 1.226393520724154),
             '10': (0.9441713925645873, 1, 1.2064272211720228)},
     "E": {'2': (1.0, 1.0, 1.0),
           '10': (1.0, 1.0, 1.0)}}


# -------------------------------------------------------------
# The conversion functions that make use of the matrices above
# -------------------------------------------------------------


##### HSV - RGB

def rgb2hsv(rgb):
    """RGB to HSV color space conversion.
    Parameters
    ----------
    rgb : tensor
        The image in RGB format, in a 3-D array of shape ``(.., .., 3)``.
    Returns
    -------
    out : ndarray
        The image in HSV format, in a 3-D array of shape ``(.., .., 3)``.
    Notes
    -----
    Conversion between RGB and HSV color spaces results in some loss of
    precision, due to integer arithmetic and rounding [1]_.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/HSL_and_HSV
    Examples
    --------
    >>> from skimage import color
    >>> from skimage import data
    >>> img = data.astronaut()
    >>> img_hsv = color.rgb2hsv(img)
    """
    arr = rgb.permute(0,2,3,1)
    out = torch.empty_like(arr)
    # -- V channel
    out_v = arr.max(3)[0]

    # -- S channel
    delta = arr.max(3)[0]-arr.min(3)[0] # np.ptp(arr, 3) ### PTP NON VA BENE
    out_s = delta / out_v
    out_s[delta == 0.] = 0.


    # -- H channel
    # red is max
    idx = (arr[:, :, :, 0] == out_v)
    out[:,:,:,0][idx] = (arr[:,:,:,1][idx] - arr[:,:,:,2][idx]) / delta[idx]

    # green is max
    idx = (arr[:,:, :, 1] == out_v)
    out[:,:,:,0][idx] = 2. + (arr[:,:,:,2][idx] - arr[:,:,:,0][idx]) / delta[idx]

    # blue is max
    idx = (arr[:,:, :, 2] == out_v)
    out[:,:,:,0][idx] = 4. + (arr[:,:,:,0][idx] - arr[:,:,:,1][idx]) / delta[idx]

    out_h = (out[:,:, :, 0] / 6.) % 1.
    out_h[delta == 0.] = 0.

    # -- output
    out[:, :, :, 0] = out_h
    out[:, :, :, 1] = out_s
    out[:, :, :, 2] = out_v

    # remove NaN

    out[torch.isnan(out)] = 0

    return out.permute(0,3,1,2)

def hsv2rgb(hsv):
    """HSV to RGB color space conversion.
    Parameters
    ----------
    hsv : array_like
        The image in HSV format, in a 3-D array of shape ``(.., 3, .., ..)``.
    Returns
    -------
    out : ndarray
        The image in RGB format, in a 3-D array of shape ``(..,3, .., ..)``.
    Notes
    -----
    Conversion between RGB and HSV color spaces results in some loss of
    precision, due to integer arithmetic and rounding [1]_.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/HSL_and_HSV
    Examples
    --------
    >>> from skimage import data
    >>> img = data.astronaut()
    >>> img_hsv = rgb2hsv(img)
    >>> img_rgb = hsv2rgb(img_hsv)
    """
    arr = hsv.permute(0,2,3,1)
    out = torch.empty_like(arr)

    hi = torch.floor(arr[:, :, :, 0] * 6)
    f = arr[:, :, :, 0] * 6 - hi
    p = arr[:, :, :, 2] * (1 - arr[:, :, :, 1])
    q = arr[:, :, :, 2] * (1 - f * arr[:, :, :, 1])
    t = arr[:, :, :, 2] * (1 - (1 - f) * arr[:, :, :, 1])
    v = arr[:, :, :, 2]

    hi = torch.stack([hi, hi, hi],3) % 6

    # import ipdb; ipdb.set_trace()
    # if hi.numpy().all()==0:
    out[:,:,:,0][hi[:,:,:,0]==0] = v[hi[:,:,:,0]==0]
    out[:,:,:,1][hi[:,:,:,1]==0] = t[hi[:,:,:,1]==0]
    out[:,:,:,2][hi[:,:,:,2]==0] = p[hi[:,:,:,2]==0]
    # if hi.numpy().all()==1:
    out[:,:,:,0][hi[:,:,:,0]==1] = q[hi[:,:,:,0]==1]
    out[:,:,:,1][hi[:,:,:,1]==1] = v[hi[:,:,:,1]==1]
    out[:,:,:,2][hi[:,:,:,2]==1] = p[hi[:,:,:,2]==1]
    # if hi.numpy().all()==2:
    out[:,:,:,0][hi[:,:,:,0]==2] = p[hi[:,:,:,0]==2]
    out[:,:,:,1][hi[:,:,:,1]==2] = v[hi[:,:,:,1]==2]
    out[:,:,:,2][hi[:,:,:,2]==2] = t[hi[:,:,:,2]==2]
    # if hi.numpy().all()==3:
    out[:,:,:,0][hi[:,:,:,0]==3] = p[hi[:,:,:,0]==3]
    out[:,:,:,1][hi[:,:,:,1]==3] = q[hi[:,:,:,1]==3]
    out[:,:,:,2][hi[:,:,:,2]==3] = v[hi[:,:,:,2]==3]
    # if hi.numpy().all()==4:
    out[:,:,:,0][hi[:,:,:,0]==4] = t[hi[:,:,:,0]==4]
    out[:,:,:,1][hi[:,:,:,1]==4] = p[hi[:,:,:,1]==4]
    out[:,:,:,2][hi[:,:,:,2]==4] = v[hi[:,:,:,2]==4]
    # if hi.numpy().all()==5:
    out[:,:,:,0][hi[:,:,:,0]==5] = v[hi[:,:,:,0]==5]
    out[:,:,:,1][hi[:,:,:,1]==5] = p[hi[:,:,:,1]==5]
    out[:,:,:,2][hi[:,:,:,2]==5] = q[hi[:,:,:,2]==5]

    return out.permute(0,3,1,2)


##### LAB - RGB

def _convert(matrix, arr):
    """Do the color space conversion.
    Parameters
    ----------
    matrix : array_like
        The 3x3 matrix to use.
    arr : array_like
        The input array.
    Returns
    -------
    out : ndarray, dtype=float
        The converted array.
    """

    bs, ch, h, w = arr.shape

    arr = arr.transpose(1,2).transpose(2,3)

    arr = arr.contiguous().view(-1,3,1)

    matrix = matrix.transpose(0,1).unsqueeze(0).cuda()
    matrix = matrix.repeat(arr.shape[0],1,1)

    res = torch.bmm(matrix, arr)

    res = res.view(bs,h,w,ch)
    res = res.transpose(3,2).transpose(2,1)

    # import ipdb; ipdb.set_trace()

    return res

def get_xyz_coords(illuminant, observer):
    """Get the XYZ coordinates of the given illuminant and observer [1]_.
    Parameters
    ----------
    illuminant : {"A", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10"}, optional
        The aperture angle of the observer.
    Returns
    -------
    (x, y, z) : tuple
        A tuple with 3 elements containing the XYZ coordinates of the given
        illuminant.
    Raises
    ------
    ValueError
        If either the illuminant or the observer angle are not supported or
        unknown.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Standard_illuminant
    """
    illuminant = illuminant.upper()
    try:
        return illuminants[illuminant][observer]
    except KeyError:
        raise ValueError("Unknown illuminant/observer combination\
        (\'{0}\', \'{1}\')".format(illuminant, observer))

def rgb2xyz(rgb):
    mask = rgb > 0.04045
    rgb[mask] = torch.pow((rgb[mask] + 0.055) / 1.055, 2.4)
    rgb[~mask] /= 12.92
    return _convert(xyz_from_rgb, rgb)

def xyz2lab(xyz, illuminant="D65", observer="2"):
    # arr = _prepare_colorarray(xyz)
    xyz_ref_white = get_xyz_coords(illuminant, observer)
    # scale by CIE XYZ tristimulus values of the reference white point
    xyz = xyz / xyz_ref_white.view(1,3,1,1).cuda()
    # Nonlinear distortion and linear transformation
    mask = xyz > 0.008856
    xyz[mask] = torch.pow(xyz[mask], 1/3)
    xyz[~mask] = 7.787 * xyz[~mask] + 16. / 116.
    x, y, z = xyz[:, 0, :, :], xyz[:, 1, :, :], xyz[:, 2, :, :]
    # Vector scaling
    L = (116. * y) - 16.
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)
    return torch.stack((L,a,b), 1)

def rgb2lab(rgb, illuminant="D65", observer="2"):
    """RGB to lab color space conversion.
    Parameters
    ----------
    rgb : array_like
        The image in RGB format, in a 3- or 4-D array of shape
        ``(.., ..,[ ..,] 3)``.
    illuminant : {"A", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10"}, optional
        The aperture angle of the observer.
    Returns
    -------
    out : ndarray
        The image in Lab format, in a 3- or 4-D array of shape
        ``(.., ..,[ ..,] 3)``.
    Raises
    ------
    ValueError
        If `rgb` is not a 3- or 4-D array of shape ``(.., ..,[ ..,] 3)``.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Standard_illuminant
    Notes
    -----
    This function uses rgb2xyz and xyz2lab.
    By default Observer= 2A, Illuminant= D65. CIE XYZ tristimulus values
    x_ref=95.047, y_ref=100., z_ref=108.883. See function `get_xyz_coords` for
    a list of supported illuminants.
    """
    return xyz2lab(rgb2xyz(rgb), illuminant, observer)





def lab2xyz(lab, illuminant="D65", observer="2"):
    arr = lab
    L, a, b = arr[:, 0, :, :], arr[:, 1, :, :], arr[:, 2, :, :]
    y = (L + 16.) / 116.
    x = (a / 500.) + y
    z = y - (b / 200.)

    if (z < 0).sum() > 0:
        invalid = np.nonzero(z < 0)
        warn('Color data out of range: Z < 0 in %s pixels' % invalid[0].size)
        z[invalid] = 0

    out = torch.stack((x, y, z),1)

    mask = out > 0.2068966
    out[mask] = torch.pow(out[mask], 3.)
    out[~mask] = (out[~mask] - 16.0 / 116.) / 7.787

    # rescale to the reference white (illuminant)
    xyz_ref_white = get_xyz_coords(illuminant, observer)
    xyz_ref_white = xyz_ref_white.unsqueeze(2).unsqueeze(2).repeat(1,1,out.shape[2],out.shape[3]).cuda()
    out *= xyz_ref_white
    return out

def xyz2rgb(xyz):
    arr = _convert(rgb_from_xyz, xyz)
    mask = arr > 0.0031308
    arr[mask] = 1.055 * torch.pow(arr[mask], 1 / 2.4) - 0.055
    arr[~mask] *= 12.92
    torch.clamp(arr, 0, 1, out=arr)
    return arr

def lab2rgb(lab, illuminant="D65", observer="2"):
    """Lab to RGB color space conversion.
    Parameters
    ----------
    lab : array_like
        The image in Lab format, in a 3-D array of shape ``(.., .., 3)``.
    illuminant : {"A", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10"}, optional
        The aperture angle of the observer.
    Returns
    -------
    out : ndarray
        The image in RGB format, in a 3-D array of shape ``(.., .., 3)``.
    Raises
    ------
    ValueError
        If `lab` is not a 3-D array of shape ``(.., .., 3)``.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Standard_illuminant
    Notes
    -----
    This function uses lab2xyz and xyz2rgb.
    By default Observer= 2A, Illuminant= D65. CIE XYZ tristimulus values
    x_ref=95.047, y_ref=100., z_ref=108.883. See function `get_xyz_coords` for
    a list of supported illuminants.
    """
    return xyz2rgb(lab2xyz(lab, illuminant, observer))

import numpy as np
from PIL import Image


def pad_reflection(image, top, bottom, left, right):
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return image
    h, w = image.shape[:2]
    next_top = next_bottom = next_left = next_right = 0
    if top > h - 1:
        next_top = top - h + 1
        top = h - 1
    if bottom > h - 1:
        next_bottom = bottom - h + 1
        bottom = h - 1
    if left > w - 1:
        next_left = left - w + 1
        left = w - 1
    if right > w - 1:
        next_right = right - w + 1
        right = w - 1
    new_shape = list(image.shape)
    new_shape[0] += top + bottom
    new_shape[1] += left + right
    new_image = np.empty(new_shape, dtype=image.dtype)
    new_image[top:top+h, left:left+w] = image
    new_image[:top, left:left+w] = image[top:0:-1, :]
    new_image[top+h:, left:left+w] = image[-1:-bottom-1:-1, :]
    new_image[:, :left] = new_image[:, left*2:left:-1]
    new_image[:, left+w:] = new_image[:, -right-1:-right*2-1:-1]
    return pad_reflection(new_image, next_top, next_bottom,
                          next_left, next_right)


def pad_constant(image, top, bottom, left, right, value):
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return image
    h, w = image.shape[:2]
    new_shape = list(image.shape)
    new_shape[0] += top + bottom
    new_shape[1] += left + right
    new_image = np.empty(new_shape, dtype=image.dtype)
    new_image.fill(value)
    new_image[top:top+h, left:left+w] = image
    return new_image


def pad_image(mode, image, top, bottom, left, right, value=0):
    if mode == 'reflection':
        return Image.fromarray(
            pad_reflection(np.asarray(image), top, bottom, left, right))
    elif mode == 'constant':
        return Image.fromarray(
            pad_constant(np.asarray(image), top, bottom, left, right, value))
    else:
        raise ValueError('Unknown mode {}'.format(mode))

class PadStride(object):
    """Pads the given image on all sides such that its size is divisible by stride"""

    def __init__(self, stride, fill=0):
        self.stride = stride
        self.fill = fill

    def __call__(self, image, label=None, *args):

        # for PIL images
        # sz = np.array(image.size)

        ## For numpy arrays
        hh, ww, _ = image.shape
        sz = np.array((ww,hh))

        pad_values = np.mod(self.stride - np.mod(sz, self.stride), self.stride)
        pad_values = np.concatenate((np.floor(pad_values/2),
                                     np.ceil(pad_values/2))).astype(np.int8)
        if label is not None:
            label = pad_image(
                'constant', label,
                pad_values[1], pad_values[3], pad_values[0], pad_values[2],
                value=255)
        if self.fill == -1:
            image = pad_image(
                'reflection', image,
                pad_values[1], pad_values[3], pad_values[0], pad_values[2])
        else:
            image = pad_image(
                'constant', image,
                pad_values[1], pad_values[3], pad_values[0], pad_values[2],
                value=self.fill)
        return image

import cv2
import numpy as np

import torch


def im2double(im):
    """ Returns a double image [0,1] of the uint im. """
    return cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

def from_tensor_to_image(tensor, device='cuda'):
    """ converts tensor to image """
    tensor = torch.squeeze(tensor, dim=0)
    if device == 'cpu':
        image = tensor.data.numpy()
    else:
        image = tensor.cpu().data.numpy()
    # CHW to HWC
    image = image.transpose((1, 2, 0))
    image = from_rgb2bgr(image)
    return image

def from_image_to_tensor(image):
    image = from_bgr2rgb(image)
    image = im2double(image)  # convert to double
    image = np.array(image)
    assert len(image.shape) == 3, 'Input image should be 3 channels colored images'
    # HWC to CHW
    image = image.transpose((2, 0, 1))
    return torch.from_numpy(image)


def from_bgr2rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB

def from_rgb2bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # convert from BGR to RGB


def load_optimizer(optimizer, path, device):
    if path != '':
        print("\n===> Loading optimizer checkpoint")
        optimizer.load_state_dict(torch.load(path))
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

def load_model(model, path):
    if path != '':
        print("\n===> Loading checkpoint")
        model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

def save_chkpt(state, path, epoch, name):
    torch.save(state, '{}/{}_epoch{}.pth'.format(path, name, str(epoch)))

def save_model(state, path, epoch, name):
    torch.save(state, '{}/net{}_epoch{}.pth'.format(path, name, str(epoch)))

def save_optim(state, path, epoch, name):
    torch.save(state, '{}/optim{}_epoch{}.pth'.format(path, name, str(epoch)))


def adjust_learning_rate(optimizer, sf=0.1):
    """Sets the learning rate to the optimizer LR decayed by sf """
    lr = optimizer.param_groups[0]['lr']
    lr = lr * sf
    print('\n\033[92m|NEW LR:\033[0m ' + str(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def lr_scheduler_setup(optimizer, epoch_number, mode='lambda1'):

    if mode == 'lambda1':
        lambda_lr = lambda epoch: 1
    elif mode == 'lambda2':
        lambda_lr = lambda epoch: pow((1-(epoch/epoch_number)),0.6)
    elif mode == 'lambda3':
        lambda_lr = lambda epoch: pow((1-(epoch/epoch_number)),0.9)
    else:
        raise NotImplementedError('scheduler [%s] is not implemented' % init_type)

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

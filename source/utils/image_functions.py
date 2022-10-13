import torch


def transform_image(im, coeff):

    bs, ch, h, w = im.shape

    image = torch.cat([im,torch.ones(bs,1,h,w).cuda()], 1)

    # import ipdb; ipdb.set_trace()
    image = image.transpose(1,2).transpose(2,3)
    image = image.contiguous().view(-1,4,1)

    coeff = coeff.transpose(1,2).transpose(2,3)
    coeff = coeff.contiguous().view(-1,3,4)


    res = torch.bmm(coeff, image)

    res = res.view(bs,h,w,ch)
    res = res.transpose(3,2).transpose(2,1)

    return res

def transform_image_gamma(im, coeff):

    bs, ch, h, w = im.shape

    image = im.clone()

    image[:,0,:,:] = im[:,0,:,:]*coeff[:,0,:,:] + coeff[:,1,:,:]
    image[:,1,:,:] = im[:,1,:,:]*coeff[:,2,:,:] + coeff[:,3,:,:]
    image[:,2,:,:] = im[:,2,:,:]*coeff[:,4,:,:] + coeff[:,5,:,:]

    return image


def third_g_hs(im, coeff):

    bs, ch, h, w = im.shape

    image = torch.cat([im,torch.ones(bs,1,h,w).cuda()], 1)

    # import ipdb; ipdb.set_trace()
    image = image.transpose(1,2).transpose(2,3)
    image = image.contiguous().view(-1,3,1)

    coeff = coeff.transpose(1,2).transpose(2,3)
    coeff = coeff.contiguous().view(-1,2,3)


    res = torch.bmm(coeff, image)

    res = res.view(bs,h,w,ch)
    res = res.transpose(3,2).transpose(2,1)

    return res

def third_g_v(im, coeff):

    im = im.unsqueeze(1)
    bs, ch, h, w = im.shape

    image = torch.cat([im**3, im**2, im, im**0], 1)

    # import ipdb; ipdb.set_trace()
    image = image.transpose(1,2).transpose(2,3)
    image = image.contiguous().view(-1,4,1)

    coeff = coeff.transpose(1,2).transpose(2,3)
    coeff = coeff.contiguous().view(-1,1,4)


    res = torch.bmm(coeff, image)

    res = res.view(bs,h,w,ch)
    res = res.transpose(3,2).transpose(2,1)

    return res

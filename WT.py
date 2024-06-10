import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

def default_conv(in_channels, out_channels, kernel_size, bias=True, dilation=1):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2)+dilation-1, bias=bias, dilation=dilation)


def default_conv1(in_channels, out_channels, kernel_size, bias=True, groups=3):
    return nn.Conv3d(
        in_channels,out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, groups=groups)



def channel_shuffle(x, groups):
    batchsize, num_channels, depth, height, width = x.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,channels_per_group,depth, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1,depth, height, width)

    return x

def pixel_down_shuffle(x, downsacale_factor):
    batchsize, num_channels, depth, height, width = x.size()

    out_depth = depth // downsacale_factor
    out_height = height // downsacale_factor
    out_width = width // downsacale_factor
    input_view = x.contiguous().view(batchsize, num_channels,
                                     out_depth, downsacale_factor,
                                     out_height, downsacale_factor,
                                     out_width, downsacale_factor, 
                                     downsacale_factor)

    num_channels *= downsacale_factor ** 3
    unshuffle_out = input_view.permute(0,1,3,5,7,2,4,6).contiguous()
    return unshuffle_out.view(batchsize, num_channels, out_depth, out_height, out_width)



def dwt3_init(x):
    x00 =  x[:, :, 0::2, 0::2, :] / 2
    x01 =  x[:, :, 0::2, 1::2, :] / 2
    x10 =  x[:, :, 1::2, 0::2, :] / 2
    x11 =  x[:, :, 1::2, 1::2, :] / 2
    
    x000 = x00[:, :, :, :, 0::2]
    x010 = x01[:, :, :, :, 0::2]
    x001 = x00[:, :, :, :, 1::2]
    x011 = x01[:, :, :, :, 1::2]

    x100 = x10[:, :, :, :, 0::2]
    x110 = x11[:, :, :, :, 0::2]
    x101 = x10[:, :, :, :, 1::2]
    x111 = x11[:, :, :, :, 1::2]

    x_LLL =  x000 + x010 + x001 + x011
    x_LHL = -x000 - x010 + x001 + x011
    x_LLH = -x000 + x010 - x001 + x011
    x_LHH =  x000 - x010 - x001 + x011

    x_HLL =  x100 + x110 + x101 + x111
    x_HHL = -x100 - x110 + x101 + x111
    x_HLH = -x100 + x110 - x101 + x111
    x_HHH =  x100 - x110 - x101 + x111

    return torch.cat((x_LLL, x_LHL, x_LLH, x_LHH, x_HLL, x_HHL, x_HLH, x_HHH), 1)

def iwt3_init(x,cpu):
    r = 2
    in_batch, in_channel, in_depth, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_depth, out_height, out_width = in_batch, int(
        in_channel / (r ** 3)), r * in_depth, r * in_height, r * in_width
    
    x01 = x[:, out_channel * 0:out_channel * 1, :, :, :] / 2
    x02 = x[:, out_channel * 1:out_channel * 2, :, :, :] / 2
    x03 = x[:, out_channel * 2:out_channel * 3, :, :, :] / 2
    x04 = x[:, out_channel * 3:out_channel * 4, :, :, :] / 2
    
    x11 = x[:, out_channel * 4:out_channel * 5, :, :, :] / 2
    x12 = x[:, out_channel * 5:out_channel * 6, :, :, :] / 2
    x13 = x[:, out_channel * 6:out_channel * 7, :, :, :] / 2
    x14 = x[:, out_channel * 7:out_channel * 8, :, :, :] / 2
    
    if cpu:
        h = torch.zeros([out_batch, out_channel, out_depth, out_height, out_width]).float()
    else:
        h = torch.zeros([out_batch, out_channel, out_depth, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2, 0::2] = x01 - x02 - x03 + x04
    h[:, :, 0::2, 1::2, 0::2] = x01 - x02 + x03 - x04
    h[:, :, 0::2, 0::2, 1::2] = x01 + x02 - x03 - x04
    h[:, :, 0::2, 1::2, 1::2] = x01 + x02 + x03 + x04
    
    h[:, :, 1::2, 0::2, 0::2] = x11 - x12 - x13 + x14
    h[:, :, 1::2, 1::2, 0::2] = x11 - x12 + x13 - x14
    h[:, :, 1::2, 0::2, 1::2] = x11 + x12 - x13 - x14
    h[:, :, 1::2, 1::2, 1::2] = x11 + x12 + x13 + x14

    return h

class Channel_Shuffle(nn.Module):
    def __init__(self, conv_groups):
        super(Channel_Shuffle, self).__init__()
        self.conv_groups = conv_groups
        self.requires_grad = False

    def forward(self, x):
        return channel_shuffle(x, self.conv_groups)

#class SP(nn.Module):
#    def __init__(self):
#        super(SP, self).__init__()
#        self.requires_grad = False

#    def forward(self, x):
#        return sp_init(x)

class Pixel_Down_Shuffle(nn.Module):
    def __init__(self):
        super(Pixel_Down_Shuffle, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return pixel_down_shuffle(x, 2)

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt3_init(x)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.cpu = False
        self.requires_grad = False

    def forward(self,x):
        return iwt3_init(x,self.cpu)





class BBlock(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(BBlock, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias))
        if bn: m.append(nn.BatchNorm3d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x).mul(self.res_scale)
        return x

class DBlock_com(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_com, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm3d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=3))
        if bn: m.append(nn.BatchNorm3d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x

class DBlock_inv(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_inv, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=3))
        if bn: m.append(nn.BatchNorm3d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm3d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x

class DBlock_com1(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_com1, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm3d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=1))
        if bn: m.append(nn.BatchNorm3d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x

class DBlock_inv1(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_inv1, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm3d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=1))
        if bn: m.append(nn.BatchNorm3d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x

class DBlock_com2(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_com2, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm3d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm3d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x

class DBlock_inv2(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_inv2, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm3d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm3d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x

class ShuffleBlock(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1,conv_groups=1):

        super(ShuffleBlock, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias))
        m.append(Channel_Shuffle(conv_groups))
        if bn: m.append(nn.BatchNorm3d(out_channels))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x).mul(self.res_scale)
        return x


class DWBlock(nn.Module):
    def __init__(
        self, conv, conv1, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DWBlock, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias))
        if bn: m.append(nn.BatchNorm3d(out_channels))
        m.append(act)

        m.append(conv1(in_channels, out_channels, 1, bias=bias))
        if bn: m.append(nn.BatchNorm3d(out_channels))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x).mul(self.res_scale)
        return x

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm3d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Block(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(Block, self).__init__()
        m = []
        for i in range(4):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm3d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        # res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 8 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm3d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 27 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm3d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
import WT as common
import torch
import torch.nn as nn
import scipy.io as sio


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=4, num_heads=1):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
    
        B, C, D, H, W = x.shape
        x = x.view(B, C, D // self.window_size, self.window_size, H // self.window_size, self.window_size, W // self.window_size, self.window_size)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1) 
        x = x.reshape(-1, self.window_size**3, C)  


        qkv = self.qkv(x).reshape(-1, self.window_size**3, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(-1, self.window_size**3, C)

 
        x = self.proj(x)
        x = x.view(B, D // self.window_size, H // self.window_size, W // self.window_size, self.window_size, self.window_size, self.window_size, -1)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, C, D, H, W)  

        return x


class Denoising_Net(nn.Module):
    def __init__(self, in_channels):
        super(Denoising_Net, self).__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.attn1 = WindowAttention(64)
        self.conv2 = ConvBlock(64, 128)
        self.attn2 = WindowAttention(128)
        self.conv3 = ConvBlock(128, 256)
        self.attn3 = WindowAttention(256)

        self.final_conv = nn.Conv3d(256, in_channels, kernel_size=3, padding=1)  # 输出通道数与输入匹配

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.attn1(x)
        x = self.conv2(x)
        x = self.attn2(x)
        x = self.conv3(x)
        x = self.attn3(x)
        x = self.final_conv(x)
        return x + residual



def make_model(args, parent=False):
    return Reconstruction_Net(args)

class Reconstruction_Net(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(Reconstruction_Net, self).__init__()
        print('Model :Reconstruction_Net')
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        self.scale_idx = 0
        nColor = args.n_colors

        act = nn.ReLU(True)

        self.DWT = common.DWT()
        self.IWT = common.IWT(args)

        n = 1
        m_head = [common.BBlock(conv, nColor, n_feats, kernel_size, act=act)]
        d_l0 = []
        d_l0.append(common.DBlock_com1(conv, n_feats, n_feats, kernel_size, act=act, bn=False))


        d_l1 = [common.BBlock(conv, n_feats * 8, n_feats * 2, kernel_size, act=act, bn=False)]
        d_l1.append(common.DBlock_com1(conv, n_feats * 2, n_feats * 2, kernel_size, act=act, bn=False))

        d_l2 = []
        d_l2.append(common.BBlock(conv, n_feats * 16, n_feats * 8, kernel_size, act=act, bn=False))
        d_l2.append(common.DBlock_com1(conv, n_feats * 8, n_feats * 8, kernel_size, act=act, bn=False))
        d_l3 = []
        d_l3.append(common.BBlock(conv, n_feats * 64, n_feats * 32, kernel_size, act=act, bn=False))
        d_l3.append(common.DBlock_com(conv, n_feats * 32, n_feats * 32, kernel_size, act=act, bn=False))

        i_l3 = [common.DBlock_inv(conv, n_feats * 32, n_feats * 32, kernel_size, act=act, bn=False)]
        i_l3.append(common.BBlock(conv, n_feats * 32, n_feats * 64, kernel_size, act=act, bn=False))

        i_l2 = [common.DBlock_inv1(conv, n_feats * 8, n_feats * 8, kernel_size, act=act, bn=False)]
        i_l2.append(common.BBlock(conv, n_feats * 8, n_feats * 16, kernel_size, act=act, bn=False))

        i_l1 = [common.DBlock_inv1(conv, n_feats * 2, n_feats * 2, kernel_size, act=act, bn=False)]
        i_l1.append(common.BBlock(conv, n_feats * 2, n_feats * 8, kernel_size, act=act, bn=False))

        i_l0 = [common.DBlock_inv1(conv, n_feats, n_feats, kernel_size, act=act, bn=False)]

        m_tail = [conv(n_feats, nColor, kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.d_l3 = nn.Sequential(*d_l3)
        self.d_l2 = nn.Sequential(*d_l2)
        self.d_l1 = nn.Sequential(*d_l1)
        self.d_l0 = nn.Sequential(*d_l0)
        self.i_l3 = nn.Sequential(*i_l3)
        self.i_l2 = nn.Sequential(*i_l2)
        self.i_l1 = nn.Sequential(*i_l1)
        self.i_l0 = nn.Sequential(*i_l0)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x0 = self.d_l0(self.head(x))
        x1 = self.d_l1(self.DWT(x0))
        x2 = self.d_l2(self.DWT(x1))
        x3 = self.d_l3(self.DWT(x2))
        x_ = self.IWT(self.i_l3(x3)) + x2
        x_ = self.IWT(self.i_l2(x_)) + x1
        x_ = self.IWT(self.i_l1(x_)) + x0
        x  = self.tail(self.i_l0(x_)) + x

        return x

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
import torch.nn as nn
import torch

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        # return super().forward(x.float()).type(x.dtype)
        return super().forward(x)
    
    
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class ZeroSFT(nn.Module):
    def __init__(self, label_nc, norm_nc, concat_channels=0, norm=False, mask=False):
        super().__init__()

        # param_free_norm_type = str(parsed.group(1))
        ks = 3
        pw = ks // 2

        self.norm = norm
        if self.norm:
            self.param_free_norm = normalization(norm_nc + concat_channels)
        else:
            self.param_free_norm = nn.Identity()
        nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.SiLU()
        )
        self.zero_mul = zero_module(nn.Conv2d(nhidden, norm_nc + concat_channels, kernel_size=ks, padding=pw))
        self.zero_add = zero_module(nn.Conv2d(nhidden, norm_nc + concat_channels, kernel_size=ks, padding=pw))
        # self.zero_mul = nn.Conv2d(nhidden, norm_nc + concat_channels, kernel_size=ks, padding=pw)
        # self.zero_add = nn.Conv2d(nhidden, norm_nc + concat_channels, kernel_size=ks, padding=pw)

        self.zero_conv = zero_module(conv_nd(2, label_nc, norm_nc, 1, 1, 0))
        self.pre_concat = bool(concat_channels != 0)
        self.mask = mask
        
        # c modulate h
        # label_nc = channels of c
        # norm_nc = channels of h

    def forward(self, c, h, h_ori=None, control_scale=1):
        assert self.mask is False
        h_raw = h
        if self.mask:
            h = h + self.zero_conv(c) * torch.zeros_like(h)
        else:
            h = h + self.zero_conv(c)
        actv = self.mlp_shared(c)
        gamma = self.zero_mul(actv)
        beta = self.zero_add(actv)
        h = self.param_free_norm(h) * (gamma + 1) + beta
        return h * control_scale + h_raw * (1 - control_scale)
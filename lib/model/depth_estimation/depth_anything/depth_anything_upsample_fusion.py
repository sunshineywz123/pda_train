from typing import List
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from hydra.utils import instantiate
from collections import OrderedDict
import functools
from lib.utils.pylogger import Log
from lib.model.depth_estimation.depth_anything.depth_anything import DepthAnything as BaseModel
import numpy as np
EPS = 1e-2

class DepthAnythingPipeline(BaseModel):
    def __init__(self,
                 range_ratio: List,
                 inference_method: int = 0,
                 encoder='vitl',
                 features=256,
                 out_channels=[256, 512, 1024, 1024],
                 use_bn=False,
                 use_clstoken=False,
                 localhub=True,
                 load_pretrain_backbone=None,
                 load_pretrain_net=None,
                 block_type='featurefusionblock',
                 l1_weight=0.,
                 normalize_disp_type='nonorm',
                 max_depth=20.,
                 loss_cfg=None,
                 depth_dropout_prob=0.0,
                 downsample_scale=7.5,
                 depth_padding=0.0,
                 mask_far_depth=-1,
                 add_grad=True,
                 grad_tag=None,
                 grad_tags=['HyperSim'],
                 scannetpp_grad_weight=0.1,
                 random_use_sparse=0.,
                 fit_to_tar=False,
                 output_act='sigmoid',
                 cspn=False,
                 warp_func=None):
        # import pdb;pdb.set_trace()
        super().__init__(encoder=encoder, features=features, out_channels=out_channels, use_bn=use_bn, use_clstoken=use_clstoken,
                         localhub=localhub, load_pretrain_backbone=load_pretrain_backbone, load_pretrain_net=load_pretrain_net, block_type=block_type,
                         output_act=output_act, warp_func=warp_func)
        self._range_ratio = range_ratio
        self._inference_method = inference_method
        self._l1_weight = l1_weight
        self._normalize_disp_type = normalize_disp_type
        self._max_depth = max_depth
        self._random_use_sparse = random_use_sparse
        self._depth_dropout_prob = depth_dropout_prob
        self._downsample_scale = downsample_scale
        self._depth_padding = depth_padding
        self._mask_far_depth = mask_far_depth
        self._add_grad = add_grad
        self._grad_tag = grad_tag
        self._grad_tags = grad_tags
        self._scannetpp_grad_weight = scannetpp_grad_weight
        self._fit_to_tar = fit_to_tar
        self._cspn = cspn
        if self._cspn:
            self.cspn_net = CSPN()
            self.cspn_net.apply(functools.partial(weights_init, mode='trunc'))
        Log.info('Using depth padding: {}'.format(self._depth_padding))
        
        if loss_cfg is not None: self.criterion = instantiate(loss_cfg)
        else: self.criterion = L1Loss()

    def get_lowres_depth(self, batch):
        if 'lowres_depth' in batch: return batch['lowres_depth']
        else: 
            import ipdb; ipdb.set_trace()
            return self.get_simdpt(batch['depth'],
                                   int(batch['image'].shape[2] // self._downsample_scale),
                                   int(batch['image'].shape[3] // self._downsample_scale),
                                   inference_method=self._inference_method if not self.training else -1,
                                   batch=batch)

    def forward_train(self, batches):
        if not isinstance(batches, List): return self.forward_train_batch(batches)
        loss = 0.
        outputs = []
        for batch in batches:
            outputs.append(self.forward_train_batch(batch))
            loss += outputs[-1]["loss"]
        return {'loss': loss}

    def forward_train_batch(self, batch):
        '''
        1. 计算depth anything的output space
        2. align 低分辨率 depth和depth anything的output space
        3. 将align后的低分辨率depth作为condition输入depth anything
        4. 计算loss
        '''
        # try: output = self.forward_test(batch, resize=False, training=True)
        # except: import ipdb; ipdb.set_trace()
        output = self.forward_test(batch, resize=False, training=True)
        
        depth = output['depth']
        gt_depth = batch['depth']
        # mesh for zipnerf
        mesh_depth = None if 'mesh_depth' not in batch else batch['mesh_depth']
        mesh_mask = None if mesh_depth is None else mesh_depth > 0
        seg = None if 'semantic' not in batch else batch['semantic']
        
        gt_depth = self.warp_func.warp(gt_depth, reference=self.get_lowres_depth(batch))
        if mesh_depth is not None:
            mesh_depth = self.warp_func.warp(mesh_depth, reference=self.get_lowres_depth(batch))
        # if 'minmax' in self._normalize_disp_type:
        #     depth_min, depth_max = output['depth_min'], output['depth_max']
        #     if isinstance(depth_min, float):
        #         gt_depth = (gt_depth - depth_min) / (depth_max - depth_min)
        #         if mesh_depth is not None:
        #             mesh_depth = (mesh_depth - depth_min) / (depth_max - depth_min)
        #     else:
        #         gt_depth = (gt_depth - depth_min[:, None, None]) / (depth_max - depth_min)[:, None, None]
        #         if mesh_depth is not None:
        #             mesh_depth = (mesh_depth - depth_min[:, None, None]) / (depth_max - depth_min)[:, None, None]
        ret_dict = {}
        add_grad = self._add_grad
        for grad_tag in self._grad_tags:
            if grad_tag.lower() in batch['image_path'][0].lower():
                add_grad = True
        scannetpp_grad_weight = self._scannetpp_grad_weight if 'scannetpp' in batch['image_path'][0].lower() else None
        mask = batch['mask']
        if np.random.random() < self._random_use_sparse and 'sparse_depth' in batch:
            mask = (batch['sparse_depth'] != 0).int()
            gt_depth = self.warp_func.warp(batch['sparse_depth'], reference=self.get_lowres_depth(batch))
        loss, loss_item = self.criterion(depth, 
                                         gt_depth, 
                                         mask, 
                                         add_grad=add_grad, 
                                         mesh_depth=mesh_depth, 
                                         mesh_mask=mesh_mask, 
                                         seg=seg,
                                         grad_weight=scannetpp_grad_weight)
        ret_dict.update(loss_item)
        ret_dict.update({'loss': loss})
        if loss.isnan().any() or loss.isinf().any():
            import ipdb; ipdb.set_trace()
        return ret_dict
        
    def forward_test(self, batch, resize=True, training=False):
        # import ipdb; ipdb.set_trace()
        lowres_depth = self.get_lowres_depth(batch)
        # import matplotlib.pyplot as plt 
        # plt.subplot(131)
        # plt.imshow(batch['image'].permute((0,2,3,1))[0].detach().cpu().numpy())
        # plt.axis('off')
        # plt.subplot(132)
        # plt.imshow(lowres_depth[0, 0].detach().cpu().numpy())
        # plt.axis('off')
        # plt.subplot(133)
        # plt.imshow(batch['depth'][0][0].detach().cpu().numpy())
        # plt.axis('off')
        # plt.tight_layout()
        # plt.savefig('test.jpg', dpi=300)
        # import ipdb; ipdb.set_trace()
        # warp depth to the same space 
        lowres_input = self.warp_func.warp(lowres_depth, reference=lowres_depth)
        # if self._mask_far_depth > 0:
        #     depth_min = 0.
        #     depth_max = self._max_depth
        #     lowres_depth = torch.clip(lowres_depth, 0., self._mask_far_depth)
        #     lowres_depth = lowres_depth / depth_max
        # else:
        #     depth_min, depth_max = lowres_depth.reshape(lowres_depth.shape[0], -1).min(1, keepdim=True)[0], lowres_depth.reshape(lowres_depth.shape[0], -1).max(1, keepdim=True)[0]
        #     if ((depth_max - depth_min) < 1e-3).any(): 
        #         depth_max[(depth_max - depth_min)<1e-3] = depth_min[(depth_max - depth_min)<1e-3] + 1e-3
                
        # lowres_input = lowres_depth # if self._normalize_disp_type == 'nonorm' else (lowres_depth, self._normalize_disp_type)
        if self.training and np.random.random() < self._depth_dropout_prob:
            lowres_input = None
            
        depth = self.forward((batch['image'] - self._mean) / self._std, lowres_input, batch=batch).unsqueeze(1)
        
        if resize and 'depth' in batch and depth.shape[2:] != batch['depth'].shape[2:]:
            depth = F.interpolate(depth, batch['depth'].shape[2:], mode='bilinear', align_corners=False)
        
        if not training:
            depth = self.warp_func.unwarp(depth, reference=lowres_depth)    
        # if not training and self._normalize_disp_type != 'nonorm':
        #     # unwarp
        #     if isinstance(depth_min, float): depth = depth * (depth_max - depth_min) + depth_min
        #     else: depth = depth * (depth_max - depth_min)[:, None, None] + depth_min[:, None, None]
        if self._fit_to_tar and 'lowres_depth' in batch: depth = self.fit_pred_to_tar(depth, batch['lowres_depth'], batch['mask'])
        return {'depth': depth} #, 'depth_min': depth_min, 'depth_max': depth_max, 'lowres_depth': lowres_depth}

    def forward(self, x, lowres_depth, need_align=False, batch=None):
        h, w = x.shape[-2:]

        layer_idxs = {'vitg': [9, 19, 29, 39], 'vitl': [
            4, 11, 17, 23], 'vitb': [2, 5, 8, 11], 'vits': [2, 5, 8, 11]}
        features = self.pretrained.get_intermediate_layers(
            x, layer_idxs[self.encoder], return_class_token=True)

        patch_h, patch_w = h // 14, w // 14
        depth, depth_feat = self.depth_head(features, patch_h, patch_w, lowres_depth, return_feat=True)
        if self._cspn and 'sparse_depth' in batch and (batch['sparse_depth'] != 0.).sum()>=10:
            sparse_depth = self.warp_func.warp(batch['sparse_depth'])
            sparse_depth[sparse_depth < 0] = 0.
            if depth.shape[2] != sparse_depth.shape[2]:
                orig_h, orig_w = sparse_depth.shape[2:]
                tar_h, tar_w = depth.shape[2:]
                uv = sparse_depth.nonzero()
                tar_uv = uv.clone()
                tar_uv[:, 2] = (uv[:, 2] * tar_h / orig_h).long()
                tar_uv[:, 3] = (uv[:, 3] * tar_w / orig_w).long()
                tar_sparse_depth = torch.zeros_like(depth)
                tar_sparse_depth[tar_uv[:, 0], tar_uv[:, 1], tar_uv[:, 2], tar_uv[:, 3]] = sparse_depth[uv[:, 0], uv[:, 1], uv[:, 2], uv[:, 3]]
                sparse_depth = tar_sparse_depth
            depth = self.cspn_net(depth_feat, depth, sparse_depth)
        if depth.shape[2:] != (h, w):
            depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = F.relu(depth)

        if need_align:
            features = features[-1][0].permute(0, 2, 1)
            b, c = features.shape[:2]
            features = features.reshape(b, c, patch_h, patch_w)
            return depth.squeeze(1), features
        
        if self._depth_padding != 0.:
            depth = depth * (1 + 2*self._depth_padding) - self._depth_padding
        return depth.squeeze(1)

    def fit_pred_to_tar(self, pred, tar, mask=None, disp=False):
        pred_ = F.interpolate(pred.detach(), tar.shape[2:], mode='nearest')
        a, b = self.linear_fit_depth(pred_, tar if not disp else 1/torch.clip(tar, 1e-3), mask)
        pred = a * pred + b
        return pred if not disp else 1/torch.clip(pred, 1e-3)
    

def bpconvlocal(feat, weights):
    B, _, H, W = feat.shape
    kernel_size = int(math.sqrt(weights.shape[1]))
    pad = kernel_size // 2 
    feat_padded = F.pad(feat, pad=(pad, pad, pad, pad), mode='replicate')
    feat_unfolded = F.unfold(feat_padded, kernel_size=kernel_size).view(B, kernel_size*kernel_size, H, W)
    weighted_sum = torch.sum(feat_unfolded * weights, dim=1, keepdim=True)
    return weighted_sum
    

class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach() & (pred.detach()>EPS) & (target.detach()>EPS)
        diff_log = torch.log(target[valid_mask!=0]) - torch.log(pred[valid_mask!=0])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() - self.lambd * torch.pow(diff_log.mean(), 2))
        return loss, {'silog_loss': loss}
    
class L1Loss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        l1_diff = F.l1_loss(pred, target, reduction='none')
        l1_diff_sum = (l1_diff * valid_mask.float()).sum(dim=(-1, -2)) 
        l1_sum = valid_mask.float().sum(dim=(-1, -2))
        loss = (l1_diff_sum / l1_sum).mean()
        return loss, {'l1_loss': loss}
    
class L1Fp16Loss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        loss = F.l1_loss(pred[valid_mask!=0], target[valid_mask!=0], reduction='mean')
        return loss, {'l1_loss': loss}


def Conv1x1(in_planes, out_planes, stride=1, bias=False, groups=1, dilation=1, padding_mode='zeros'):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


def Conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, padding_mode='zeros', bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, padding_mode=padding_mode, groups=groups, bias=bias, dilation=dilation)

class Basic2d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None, kernel_size=3, padding=1, padding_mode='zeros',
                 act=nn.ReLU, stride=1):
        super().__init__()
        if norm_layer:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding, bias=False, padding_mode=padding_mode)
        else:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding, bias=True, padding_mode=padding_mode)
        self.conv = nn.Sequential(OrderedDict([('conv', conv)]))
        if norm_layer:
            self.conv.add_module('bn', norm_layer(out_channels))
        self.conv.add_module('relu', act())

    def forward(self, x):
        out = self.conv(x)
        return out

class GenKernel(nn.Module):
    def __init__(self, in_channels, pk, norm_layer=nn.BatchNorm2d, act=nn.ReLU, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.conv = nn.Sequential(
            Basic2d(in_channels, in_channels, norm_layer=norm_layer, act=act),
            Basic2d(in_channels, pk * pk - 1, norm_layer=norm_layer, act=act),
        )

    def forward(self, fout):
        weight = self.conv(fout)
        weight_sum = torch.sum(weight.abs(), dim=1, keepdim=True)
        weight = torch.div(weight, weight_sum + self.eps)
        weight_mid = 1 - torch.sum(weight, dim=1, keepdim=True)
        weight_pre, weight_post = torch.split(weight, [weight.shape[1] // 2, weight.shape[1] // 2], dim=1)
        weight = torch.cat([weight_pre, weight_mid, weight_post], dim=1).contiguous()
        return weight
    
class CSPN(nn.Module):
    """
    implementation of CSPN++
    """

    def __init__(self, in_channels=32, pt=12, norm_layer=nn.BatchNorm2d, act=nn.ReLU, eps=1e-6):
        super().__init__()
        self.pt = pt
        self.weight3x3 = GenKernel(in_channels, 3, norm_layer=norm_layer, act=act, eps=eps)
        self.weight5x5 = GenKernel(in_channels, 5, norm_layer=norm_layer, act=act, eps=eps)
        self.weight7x7 = GenKernel(in_channels, 7, norm_layer=norm_layer, act=act, eps=eps)
        self.convmask = nn.Sequential(
            Basic2d(in_channels, in_channels, norm_layer=norm_layer, act=act),
            Basic2d(in_channels, 3, norm_layer=None, act=nn.Sigmoid),
        )
        self.convck = nn.Sequential(
            Basic2d(in_channels, in_channels, norm_layer=norm_layer, act=act),
            Basic2d(in_channels, 3, norm_layer=None, act=functools.partial(nn.Softmax, dim=1)),
        )
        self.convct = nn.Sequential(
            Basic2d(in_channels + 3, in_channels, norm_layer=norm_layer, act=act),
            Basic2d(in_channels, 3, norm_layer=None, act=functools.partial(nn.Softmax, dim=1)),
        )

    # @custom_fwd(cast_inputs=torch.float32)
    def forward(self, fout, hn, h0):
        weight3x3 = self.weight3x3(fout)
        weight5x5 = self.weight5x5(fout)
        weight7x7 = self.weight7x7(fout)
        mask3x3, mask5x5, mask7x7 = torch.split(self.convmask(fout) * (h0 > 1e-3).float(), 1, dim=1)
        conf3x3, conf5x5, conf7x7 = torch.split(self.convck(fout), 1, dim=1)
        hn3x3 = hn5x5 = hn7x7 = hn
        hns = [hn, ]
        for i in range(self.pt):
            hn3x3 = (1. - mask3x3) * bpconvlocal(hn3x3, weight3x3) + mask3x3 * h0
            hn5x5 = (1. - mask5x5) * bpconvlocal(hn5x5, weight5x5) + mask5x5 * h0
            hn7x7 = (1. - mask7x7) * bpconvlocal(hn7x7, weight7x7) + mask7x7 * h0
            if i == self.pt // 2 - 1:
                hns.append(conf3x3 * hn3x3 + conf5x5 * hn5x5 + conf7x7 * hn7x7)
        hns.append(conf3x3 * hn3x3 + conf5x5 * hn5x5 + conf7x7 * hn7x7)
        hns = torch.cat(hns, dim=1)
        wt = self.convct(torch.cat([fout, hns], dim=1))
        hn = torch.sum(wt * hns, dim=1, keepdim=True)
        return hn
import math 
def weights_init(m, mode='trunc'):
    from torch.nn.init import _calculate_fan_in_and_fan_out
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        if hasattr(m, 'weight'):
            if mode == 'trunc':
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight.data)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                torch.nn.init.trunc_normal_(m.weight.data, mean=0, std=std)
            elif mode == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data)
            else:
                raise ValueError(f'unknown mode = {mode}')
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    if classname.find('Conv1d') != -1:
        if hasattr(m, 'weight'):
            if mode == 'trunc':
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight.data)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                torch.nn.init.trunc_normal_(m.weight.data, mean=0, std=std)
            elif mode == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data)
            else:
                raise ValueError(f'unknown mode = {mode}')
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        if mode == 'trunc':
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight.data)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            torch.nn.init.trunc_normal_(m.weight.data, mean=0, std=std)
        elif mode == 'xavier':
            torch.nn.init.xavier_normal_(m.weight.data)
        else:
            raise ValueError(f'unknown mode = {mode}')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)



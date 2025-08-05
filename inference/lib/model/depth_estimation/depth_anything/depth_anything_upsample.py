from typing import List
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from hydra.utils import instantiate
from .depth_anything import DepthAnything as BaseModel
import numpy as np

class DepthAnythingPipeline(BaseModel):
    def __init__(self, 
                 zerosft: DictConfig,
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
                 control_dpt_head=False
                 ):
        super().__init__(encoder=encoder, features=features, out_channels=out_channels, use_bn=use_bn, use_clstoken=use_clstoken, 
                         localhub=localhub, load_pretrain_backbone=load_pretrain_backbone, load_pretrain_net=load_pretrain_net)
        self.zerosft = instantiate(zerosft)
        self._range_ratio = range_ratio
        self._inference_method = inference_method
        

    def get_simdpt(self, dpt, h, w, inference_method=-1, batch=None):
        """
        计算simulated lowres depth
        dpt: (B, 1, H, W)
        h: int, height of rgb height / 8
        w: int, width of rgb width / 8
        """
        if self.training and hasattr(self, 'noise_model') and self.noise_model is not None: 
            with torch.no_grad():
                dpt = self.noise_model.get_noisy_dpt(dpt, batch)
        if inference_method == -1:  # training mode
            simulate_ratio_method = np.random.choice(
                [0, 1, 2], p=self._range_ratio)
        else:
            simulate_ratio_method = inference_method

        if simulate_ratio_method == 0:
            tar_h, tar_w = h, w
        elif simulate_ratio_method == 1:
            tar_h, tar_w = 192, 256
        elif simulate_ratio_method == 2:
            # tar_h, tar_w均匀从 (h, w) 到 (192, 256), 均为整数，h:w=3:4
            # 找到h到192之间可以被3整除的数字，随机采样
            # tar_h = np.random.choice([i for i in range(h, 192) if i % 3 == 0])
            start, end = h if h % 3 == 0 else h + (3 - h % 3), 192
            tar_h = np.random.choice(np.arange(start, end, 3))
            tar_w = int(tar_h * 4 / 3)
        else:
            raise ValueError(f"Invalid inference method: {inference_method}")
        dpt_h = dpt.shape[2]
        tar_h, tar_w = int(tar_h), int(tar_w)
        if dpt_h != tar_h:
            dpt = F.interpolate(dpt, (tar_h, tar_w), mode='bilinear', align_corners=False)
        return dpt

    def get_lowres_depth(self, batch):
        if 'lowres_depth' in batch: return batch['lowres_depth']
        else: return self.get_simdpt(batch['depth'], 
                                     batch['image'].shape[2] // 8, 
                                     batch['image'].shape[3] // 8, 
                                     inference_method=self._inference_method if not self.training else -1,
                                     batch=batch)        
    
    def get_condition(self, batch):
        import ipdb; ipdb.set_trace()
        with torch.no_grad():
            disparity = super().forward((batch['image'] - self._mean) / self._std).unsqueeze(1)
        lowres_dpt = self.get_lowres_depth(batch)
        lowres_disparity = F.interpolate(disparity, lowres_dpt.shape[2:], mode='nearest')
        a, b = self.linear_fit_depth(1/torch.clip(lowres_dpt, 1e-3), lowres_disparity)
        cond = a * (1/torch.clip(lowres_dpt, 1e-3)) + b
        cond = F.interpolate(cond, batch['image'].shape[2:], mode='bilinear', align_corners=False)
        return cond
        
    def linear_fit_depth(self, depth, lowres_depth, mask=None):
        """
        depth: high resolution depth
        lowres_depth: low resolution depth
        """
        a_list, b_list = [], []
        for b in range(len(depth)):
            msk = (depth[b] > 1e-3) & (lowres_depth[b] > 1e-3)
            if mask is not None: msk &= mask[b]
            depth_msk = depth[b][msk]
            lowres_depth_msk = lowres_depth[b][msk]
            X = torch.vstack((depth_msk, torch.ones_like(depth_msk))).transpose(0, 1) 
            solution = torch.linalg.lstsq(X.float(), lowres_depth_msk.float()).solution
            a_list.append(solution[0].item())
            b_list.append(solution[1].item())
        return torch.tensor(a_list, dtype=depth.dtype, device=depth.device)[:, None, None, None], torch.tensor(b_list, dtype=depth.dtype, device=depth.device)[:, None, None, None]
                
        
    def forward_train(self, batch):
        '''
        1. 计算depth anything的output space
        2. align 低分辨率 depth和depth anything的output space
        3. 将align后的低分辨率depth作为condition输入depth anything
        4. 计算loss
        '''
        # 1.
        import ipdb; ipdb.set_trace()
        disparity = super().forward((batch['image'] - self._mean) / self._std).unsqueeze(1)
        disparity_detach = disparity.detach()
        
        # 2. 
        lowres_dpt = self.get_lowres_depth(batch)
        lowres_msk = F.interpolate(batch['mask'].float(), lowres_dpt.shape[2:], mode='nearest') > 0.5
        lowres_disparity = F.interpolate(disparity_detach, lowres_dpt.shape[2:], mode='nearest')
        a, b = self.linear_fit_depth(1/torch.clip(lowres_dpt, 1e-3), lowres_disparity, lowres_msk)
        lowres_dpt_align = a * (1/torch.clip(lowres_dpt, 1e-3)) + b
        lowres_dpt_align = F.interpolate(lowres_dpt_align, batch['image'].shape[2:], mode='bilinear', align_corners=False)
        
        # 3.
        depth_input = self.zerosft(lowres_dpt_align, (batch['image'] - self._mean) / self._std)
        disparity_super = super().forward(depth_input).unsqueeze(1)
        
        # 4.
        # 计算gt disparity
        disparity_gt = batch['depth']
        disparity_gt[batch['mask'] == 1] = 1 / disparity_gt[batch['mask'] == 1]
        for b in range(len(disparity_gt)):
            msk = batch['mask'][b] == 1
            if msk.sum() == 0: 
                continue
            disparity_gt_min = disparity_gt[b][msk].min()
            disparity_gt_max = disparity_gt[b][msk].max()
            if disparity_gt_max - disparity_gt_min < 1e-6: disparity_gt_max = disparity_gt_min + 1e-6
            disparity_gt[b][msk] = (disparity_gt[b][msk] - disparity_gt_min) / (disparity_gt_max - disparity_gt_min)
        # 计算loss        
        loss_disparity = self.loss(disparity.squeeze(1), disparity_gt.squeeze(1), (batch['mask']==1).squeeze(1).float())
        loss_disparity_super = self.loss(disparity_super.squeeze(1), disparity_gt.squeeze(1), (batch['mask']==1).squeeze(1).float())
        loss = loss_disparity * 0.1 + loss_disparity_super
        return {'loss': loss, 'loss_disparity': loss_disparity, 'loss_disparity_super': loss_disparity_super}
        # return {'loss': loss_disparity}
            
        
    def forward_test(self, batch):
        disparity = super().forward((batch['image'] - self._mean) / self._std).unsqueeze(1)
        disparity_detach = disparity.detach()
        # 2. 
        lowres_dpt = self.get_lowres_depth(batch)
        lowres_msk = F.interpolate(batch['mask'].float(), lowres_dpt.shape[2:], mode='nearest') > 0.5
        lowres_disparity = F.interpolate(disparity_detach, lowres_dpt.shape[2:], mode='nearest')
        a, b = self.linear_fit_depth(1/torch.clip(lowres_dpt, 1e-3), lowres_disparity, lowres_msk)
        lowres_dpt_align = a * (1/torch.clip(lowres_dpt, 1e-3)) + b
        lowres_dpt_align = F.interpolate(lowres_dpt_align, batch['image'].shape[2:], mode='bilinear', align_corners=False)
        
        # 3.
        depth_input = self.zerosft(lowres_dpt_align, (batch['image'] - self._mean) / self._std)
        disparity_super = super().forward(depth_input).unsqueeze(1)
        depth = F.interpolate(disparity_super, batch['depth'].shape[2:], mode='bilinear', align_corners=False)
        rgb_depth = F.interpolate(disparity, batch['depth'].shape[2:], mode='bilinear', align_corners=False)
        
        depth = self.fit_pred_to_tar(depth, batch['lowres_depth'], disp=True)
        return {'depth': depth, 'disp': True, 'rgb_depth': rgb_depth}
        
        
    def fit_pred_to_tar(self, pred, tar, disp=False):
        pred_ = F.interpolate(pred, tar.shape[2:], mode='nearest')
        a, b = self.linear_fit_depth(pred_, tar if not disp else 1/torch.clip(tar, 1e-3))
        pred = a * pred + b
        return pred if not disp else 1/pred
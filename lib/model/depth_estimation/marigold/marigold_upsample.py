from typing import List
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from hydra.utils import instantiate
from lib.utils.pylogger import Log
from tqdm.auto import tqdm
import numpy as np

from diffusers.schedulers import DDPMScheduler, DDIMScheduler
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from transformers import CLIPTextModel, CLIPTokenizer
# from lib.model.supermotion.utils.motion3d_endecoder import EnDecoderBase

from lib.model.marigold.marigold_utils import ensemble_depths
from .marigold import MarigoldPipeline


class MarigoldUpsamplePipeline(MarigoldPipeline):
    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215

    def __init__(self, unet_opt, vae_opt, textencoder_opt, scheduler_opt_train, scheduler_opt_test, tokenizer_opt,
                 zerosft: DictConfig,
                 range_ratio: List,
                 noise_model: DictConfig = None,
                 inference_method: int = 0,  # 0: 8x, 1: 192x256, 2: 3:4
                 min_percentile=0.,
                 max_percentile=1.,
                 add_cone_noise=False,
                 cone_noise_std=1.,
                 std_random_max=False,
                 orig_mean=False,
                 **kwargs):
        """
        Args:
            args: pipeline
        """
        super().__init__(unet_opt, vae_opt, textencoder_opt, scheduler_opt_train,
                         scheduler_opt_test, tokenizer_opt, min_percentile, max_percentile, **kwargs)
        self.zerosft = instantiate(zerosft)
        if noise_model is not None: 
            self.noise_model = instantiate(noise_model, _recursive_=False)
            try:
                self.noise_model.eval()
                self.noise_model.requires_grad_(False)
            except:
                Log.info("Noise model is not set.")
                pass
        self._range_ratio = range_ratio
        self._inference_method = inference_method
        self._add_cone_noise = add_cone_noise
        self._cone_noise_std = cone_noise_std
        self._orig_mean = orig_mean
        self._std_random_max = std_random_max

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
                [0, 1, 2, 3], p=self._range_ratio)
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
        elif simulate_ratio_method == 3:
            rand_v = np.random.random()
            tar_h = int((rand_v + 1) * h)
            tar_w = int((rand_v + 1) * w)
        else:
            raise ValueError(f"Invalid inference method: {inference_method}")
        dpt_h = dpt.shape[2]
        tar_h, tar_w = int(tar_h), int(tar_w)
        if dpt_h != tar_h:
            if self._add_cone_noise and self.training and np.random.random() > 0.75:
                _, dpt_min, dpt_max = self.normalized_depth(dpt)
                scale = int(dpt_h / tar_h)
                assert scale == 8
                depth_unfold = dpt.unfold(2, scale, scale).unfold(3, scale, scale)
                mean = depth_unfold.mean(dim=(-1, -2), keepdim=False)
                orig_mean = F.interpolate(dpt, (tar_h, tar_w), mode='bilinear', align_corners=False)
                std = depth_unfold.std(dim=(-1, -2), keepdim=False)
                if self._std_random_max:
                    std = std * np.random.random()
                if self._orig_mean:
                    dpt = orig_mean + std * torch.randn_like(mean) * self._cone_noise_std
                else:
                    dpt = mean + std * torch.randn_like(mean) * self._cone_noise_std
                dpt = torch.clip(dpt, dpt_min, dpt_max)
            else:
                dpt = F.interpolate(dpt, (tar_h, tar_w), mode='bilinear', align_corners=False)
        return dpt

    def get_lowres_depth(self, batch):
        if 'lowres_depth' in batch: return batch['lowres_depth']
        else: return self.get_simdpt(batch['depth'], 
                                     batch['image'].shape[2] // 8, 
                                     batch['image'].shape[3] // 8, 
                                     inference_method=self._inference_method if not self.training else -1,
                                     batch=batch)

    def compute_cond(self, batch):
        rgb_latent = super().compute_cond(batch)
        h_8, w_8 = rgb_latent.shape[2:]
        h, w = h_8*8, w_8*8
        
        lowres_dpt = self.get_lowres_depth(batch)
        lowres_dpt, dpt_min, dpt_max = self.normalized_depth(lowres_dpt)
        
        h_low, w_low = lowres_dpt.shape[2:]
        dpt_up = F.interpolate(lowres_dpt, (h, w), mode='bilinear', align_corners=False)
        dpt_up_latent = self.encode_depth(dpt_up * 2 - 1.)
        
        if h_low != h_8: dpt_down = F.interpolate(lowres_dpt, (h_8, w_8), mode='nearest')
        else: dpt_down = lowres_dpt
        
        cond = self.zerosft(torch.cat([dpt_up_latent, dpt_down], dim=1), rgb_latent)
        return cond

    def forward_test(self, batch, num_inference_steps=10):
        output = super().forward_test(batch, num_inference_steps)
        lowres_depth = self.get_lowres_depth(batch)
        if 'confidence' in batch:
            lowres_mask = batch['confidence'] == 2
            if (lowres_mask.reshape(len(lowres_mask), -1).sum(dim=-1) == 0).any():
                Log.info('confidence == 2 but no valid pixels')
                lowres_mask = batch['confidence'] == 1
            if (lowres_mask.reshape(len(lowres_mask), -1).sum(dim=-1) == 0).any():
                Log.info('confidence == 1 but no valid pixels')
                lowres_mask = batch['confidence'] == 0
        else:
            lowres_mask = F.interpolate(batch['mask'].float(), lowres_depth.shape[2:], mode='nearest') > 0.5
        depth = self.fit_pred_to_tar(output['depth'], lowres_depth, lowres_mask, disp=False)
        return {'depth': depth}
    
    def fit_pred_to_tar(self, pred, tar, mask=None, disp=False):
        pred_ = F.interpolate(pred.detach(), tar.shape[2:], mode='nearest')
        a, b = self.linear_fit_depth(
            pred_, tar if not disp else 1/torch.clip(tar, 1e-3), mask)
        pred = a * pred + b
        return pred if not disp else 1/torch.clip(pred, 1e-3)
    
    def linear_fit_depth(self, depth, lowres_depth, mask):
        """
        depth: high resolution depth
        lowres_depth: low resolution depth
        """
        a_list, b_list = [], []
        for b in range(len(depth)):
            msk = (depth[b] > 1e-3) & (lowres_depth[b] > 1e-3) & mask[b]
            depth_msk = depth[b][msk]
            lowres_depth_msk = lowres_depth[b][msk]
            X = torch.vstack((depth_msk, torch.ones_like(depth_msk))).transpose(0, 1) 
            solution = torch.linalg.lstsq(X.float(), lowres_depth_msk.float()).solution
            a_list.append(solution[0].item())
            b_list.append(solution[1].item())
        return torch.tensor(a_list, dtype=depth.dtype, device=depth.device)[None, None, None], torch.tensor(b_list, dtype=depth.dtype, device=depth.device)[None, None, None]
            
            
        # depth = depth.flatten()
        # lowres_depth = lowres_depth.flatten()
        # mask = (depth > 0) & (lowres_depth > 0)
        # import ipdb; ipdb.set_trace()
        # # a, b = np.polyfit(lowres_depth[mask], depth[mask], 1)
        # return a, b

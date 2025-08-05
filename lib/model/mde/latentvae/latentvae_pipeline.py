import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from hydra.utils import instantiate
from lib.utils.pylogger import Log
from tqdm.auto import tqdm
import numpy as np
import hydra

from lib.utils.diffusion.pipeline_helper import PipelineHelper
from .models import DPT_DINOv2
from diffusers import AutoencoderKL

class Pipeline(nn.Module, PipelineHelper):
    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215
    def __init__(self, dpt_args, loss_args, vae_path, **kwargs):
        """
        Args:
            args: pipeline
            args_clip: clip
            args_denoiser3d: denoiser3d network
        """
        super().__init__()
        # self.model = DPTDepthModel(path='/home/linhaotong/.cache/torch/hub/checkpoints/dpt_hybrid_nyu-2ce69ec7.pt',
        #                            scale=0.000305,
        #                            shift=0.1378,
        #                            invert=True)
        self.model = DPT_DINOv2(**dpt_args)
        self.loss = hydra.utils.instantiate(loss_args)
        self.vae = AutoencoderKL.from_pretrained(vae_path)
        
        self.freeze_nets()
        
    def freeze_nets(self):
        self.vae.eval()
        self.vae.requires_grad_(False)

    def forward_train(self, batch):
        depth_latent = self.model(batch['rgb'])
        outputs = dict()
        
        gt_dpt = batch['dpt'] 
        gt_dpt = torch.clip(gt_dpt, 0.5, 80.)
        gt_dpt = 1 / gt_dpt # 1/80 - 2.
        gt_dpt = gt_dpt - 1.
        gt_dpt_latent = self.encode_depth(gt_dpt)
        
        loss = F.mse_loss(depth_latent, gt_dpt_latent, reduction='mean')
        # loss = self.loss(dpt, batch['dpt'], batch['msk'])
        outputs["loss"] = loss
        return outputs
    
    def forward_test(self, batch):
        depth_latent = self.model(batch['rgb'])
        depth = self.decode_depth(depth_latent)
        depth = torch.clip(depth, -1., 1.)
        depth = (depth + 1.) * 0.5 * 2
        
        depth = torch.clip(depth, 1/80., 1)
        depth = 1 / depth # 0.5 - inf
        return {'dpt': depth}
    
    def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        # scale latent
        depth_latent = depth_latent / self.depth_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(depth_latent)
        # z, depth_latent: (5, 4, H/8, W/8)
        stacked = self.vae.decoder(z)
        # stacked: (5, 3, H, W)
        # mean of output channels
        depth_mean = stacked.mean(dim=1, keepdim=True) # 三通道取均值
        return depth_mean
    
    def encode_depth(self, depth_in: torch.Tensor) -> torch.Tensor:
        depth_in = depth_in.repeat(1, 3, 1, 1)
        h = self.vae.encoder(depth_in)
        # h: (5, 8, H/8, W/8)
        moments = self.vae.quant_conv(h)
        # moments shape = h
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # 4, 4
        # scale latent
        rgb_latent = mean * self.depth_latent_scale_factor
        return rgb_latent
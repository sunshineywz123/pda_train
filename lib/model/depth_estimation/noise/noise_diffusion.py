from typing import Any, Dict, List
import hydra
from omegaconf import DictConfig
from sklearn import metrics
import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
from lib.utils.pylogger import Log
from os.path import join
import os
import imageio
import numpy as np
import cv2
import json
from torch.optim import AdamW, Adam

from lib.utils.dpt.eval_utils import recover_metric_depth, evaluate_rel_err, recover_metric_depth_lowres, recover_metric_depth_ransac
from einops import rearrange
import torch.nn.functional as F


class NoiseDiffusion(pl.LightningModule):
    def __init__(
        self,
        pipeline=None,
        ckpt_path=None):
        super().__init__()
        self.pipeline = instantiate(pipeline, _recursive_=False)
        self.ignored_weights_prefix = ["pipeline.text_encoder", "pipeline.vae"]
        self.load_pretrained_model(ckpt_path, ckpt_type=None)
        
    def get_min_max(self, dpt):
        B = len(dpt)
        dpt_min = torch.quantile(dpt.view(B, -1), 0.02, dim=-1).view(B, 1, 1, 1)
        dpt_max = torch.quantile(dpt.view(B, -1), 0.98, dim=-1).view(B, 1, 1, 1)
        return dpt_min, dpt_max
    
    def get_noisy_dpt(self, dpt, batch):
        """Task-specific validation step. CAP or GEN."""
        dpt_min, dpt_max = self.get_min_max(dpt)
        if ((dpt_max - dpt_min) < 1e-3).any(): return dpt
        rgb = F.interpolate(batch['image'], (192, 256), mode='bilinear', align_corners=False)
        lowres_dpt = F.interpolate(dpt, (192, 256), mode='nearest')
        output = self.pipeline.forward_test({'rgb': rgb, 'lowres_dpt': lowres_dpt}, num_inference_steps=10)
        if output['err'].isnan().any() or output['err'].isinf().any(): return dpt
        noisy_dpt = lowres_dpt - torch.clip(output['err'], -100., 100.)
        return noisy_dpt

    def load_pretrained_model(self, ckpt_path, ckpt_type):
        """Load pretrained checkpoint, and assign each weight to the corresponding part."""
        Log.info(f"Loading ckpt type `{ckpt_type}': {ckpt_path}")
        state_dict = torch.load(ckpt_path, "cpu")["state_dict"]
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        real_missing = []
        for k in missing:
            miss = True
            for ig_keys in self.ignored_weights_prefix:
                if k.startswith(ig_keys):
                    miss = False
            if miss:
                real_missing.append(k)
        if len(real_missing) > 0:
            Log.warn(f"Missing keys: {real_missing}")
        if len(unexpected) > 0:
            Log.error(f"Unexpected keys: {unexpected}")
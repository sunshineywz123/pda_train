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


class NoiseGaussian(pl.LightningModule):
    def __init__(
        self,
        mean=0,
        std=0.05,
        **kwargs):
        super().__init__()
        self._mean = mean
        self._std = std
        
    def get_min_max(self, dpt):
        B = len(dpt)
        dpt_min = torch.quantile(dpt.view(B, -1), 0.02, dim=-1).view(B, 1, 1, 1)
        dpt_max = torch.quantile(dpt.view(B, -1), 0.98, dim=-1).view(B, 1, 1, 1)
        return dpt_min, dpt_max
    
    def get_noisy_dpt(self, dpt, batch):
        dpt_min, dpt_max = self.get_min_max(dpt)
        if ((dpt_max - dpt_min) < 1e-3).any():
            gaussian_noise = 0
        else:
            gaussian_noise = torch.randn_like(dpt) * self._std + self._mean
        return dpt + gaussian_noise
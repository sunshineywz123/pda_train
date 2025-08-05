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
from .models import DPTDepthModel

class DPTPipeline(nn.Module, PipelineHelper):
    def __init__(self, dpt_args, loss_args, **kwargs):
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
        self.model = DPTDepthModel(path=dpt_args.get('path', None),
                                   scale=dpt_args.get('scale', 1.),
                                   shift=dpt_args.get('shift', 0.),
                                   invert=dpt_args.get('invert', True))
        self.loss = hydra.utils.instantiate(loss_args)

    def forward_train(self, batch):
        dpt = self.model(batch['rgb'])[:, None]
        outputs = dict()
        loss = self.loss(dpt, batch['dpt'], batch['msk'])
        outputs["loss"] = loss
        return outputs
    
    def forward_test(self, batch):
        dpt = self.model(batch['rgb'])[:, None]
        return {'dpt': dpt}
    
       
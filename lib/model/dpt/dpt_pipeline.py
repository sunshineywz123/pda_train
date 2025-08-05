import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from hydra.utils import instantiate
from lib.utils.pylogger import Log
from tqdm.auto import tqdm
import numpy as np

from lib.utils.diffusion.pipeline_helper import PipelineHelper
from .models import DPTDepthModel
from .midas_loss import ScaleAndShiftInvariantLoss
from .zoe_loss import SILogLoss



from lib.model.marigold.marigold_utils import ensemble_depths

class DPTPipeline(nn.Module, PipelineHelper):
    def __init__(self, args, **kwargs):
        """
        Args:
            args: pipeline
            args_clip: clip
            args_denoiser3d: denoiser3d network
        """
        super().__init__()
        self.args = args
        # self.model = DPTDepthModel(path='/home/linhaotong/.cache/torch/hub/checkpoints/dpt_hybrid_nyu-2ce69ec7.pt',
        self.model = DPTDepthModel(path=args.get('path', None),
                                   scale=args.get('scale', 1.),
                                   shift=args.get('shift', 0.),
                                   invert=args.get('invert', True))
        self.midas_loss = ScaleAndShiftInvariantLoss()
        self.zoe_loss = SILogLoss()
        
        # ----- Freeze ----- #
        self.freeze_nets()

    def freeze_nets(self):
        pass

    # ========== Training ========== #

    def forward_train(self, batch):
        dpt = self.model(batch['rgb'])[:, None]
        outputs = dict()
        if self.args.get('loss', 'mse') == 'mse':
            loss = F.mse_loss(dpt, batch['dpt'], reduction="mean")
        elif self.args.get('loss', 'mse') == 'zoe':
            loss = self.zoe_loss(dpt, batch['dpt'])
        elif self.args.get('loss', 'mse') == 'midas':
            loss = self.midas_loss(dpt[:, 0], batch['dpt'][:, 0], torch.ones_like(batch['dpt'][:, 0]))
        else:
            import ipdb; ipdb.set_trace()
        outputs["loss"] = loss
        return outputs
    
    def forward_test(self, batch, num_inference_steps=10, show_pbar=False):
        dpt = self.model(batch['rgb'])[:, None]
        return {'dpt': dpt}
    
       
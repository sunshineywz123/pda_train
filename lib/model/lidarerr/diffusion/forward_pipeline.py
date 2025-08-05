from re import M
from typing import List
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
from lib.utils.diffusion.pipeline_helper import PipelineHelper
# from lib.model.supermotion.utils.motion3d_endecoder import EnDecoderBase

from lib.model.marigold.marigold_utils import ensemble_depths

class DiffusionPipeline(nn.Module, PipelineHelper):
    def __init__(self, args, **kwargs):
        """
        Args:
            args: pipeline
        """
        super().__init__()
        self.args = args
        self.predict_dpt = args.get('predict_dpt', False) # 预测depth
        self.only_dpt = args.get('only_dpt', False) # 仅预测depth
        self.normal_dpt = args.get('normal_dpt', False) # 预测归一化后的depth
        
        Log.debug(args)
        
        # self.tr_scheduler = DDPMScheduler(**args.scheduler_opt_train)
        # self.te_scheduler = DDIMScheduler(**args.scheduler_opt_test)
        if self.args.get('hydra_unet', False):
            from lib.network.ivid.adm import AdmUnet2d
            self.unet = AdmUnet2d(**args.ivid_unet_opt)
        else:
            self.unet = UNet2DConditionModel(**args.unet_opt)
        # self.text_encoder = CLIPTextModel.from_pretrained(args.textencoder_opt.config_dir)
        # self.tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_opt.config_dir)
        # self.empty_text_embed = None

        # ----- Freeze ----- #
        self.freeze_nets()

    def freeze_nets(self):
        pass
        # self.text_encoder.eval()
        # self.text_encoder.requires_grad_(False)

    # ========== Training ========== #
    def __encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.text_encoder.dtype).detach()
        
        
    def forward_train(self, batches):
        if not isinstance(batches, List):
            return self.forward_train_batch(batches)
        loss = 0.
        outputs = []
        for batch in batches:
            outputs.append(self.forward_train_batch(batch))
            loss += outputs[-1]["loss"]
        return {'loss': loss}
    
    def normalized_depth(self, dpt):
        dpt_min = dpt.min(-1, keepdim=True)[0].min(-2, keepdim=True)[0]
        dpt_max = dpt.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
        msk = (dpt_max - dpt_min) < 1e-6
        if msk.any(): dpt_max[msk] = dpt_min[msk] + 1e-6
        return (dpt - dpt_min) / (dpt_max - dpt_min), dpt_min, dpt_max
        
    def forward_train_batch(self, batch):
        outputs = dict()
        scheduler = self.tr_scheduler
        # normalize gt and compute clean target
        # augmentation
        
        # normalize depth
        # compute condition
        # dpt, dpt_min, dpt_max = self.normalized_depth(batch['lowres_dpt'])
        # cond = torch.cat([batch['rgb'], dpt], dim=1)
        cond = batch['rgb']
        
        if self.predict_dpt:
            err_latent = torch.cat([batch['err'], batch['dpt'], batch['lowres_dpt']], dim=1)
        elif self.only_dpt: 
            if self.normal_dpt: err_latent, _, _ = self.normalized_depth(batch['lowres_dpt'])
            else: err_latent = batch['lowres_dpt']
        else: err_latent = batch['err']# * (dpt_max - dpt_min)
        B = len(err_latent)
        # add noise to gt
        noise = torch.randn_like(err_latent)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=err_latent.device).long()
        noisy_err_latent = scheduler.add_noise(err_latent, noise, t)

        # Encode CLIP embedding
        if self.empty_text_embed is None:
            self.__encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (B, 1, 1)
        )
        
        # *. Denoise
        unet_input = torch.cat([cond, noisy_err_latent], dim=1)
        if self.args.get('hydra_unet', False):
            target_pred = self.unet(unet_input, t, encoder_hidden_states=batch_empty_text_embed)
        else: target_pred = self.unet(unet_input, t, encoder_hidden_states=batch_empty_text_embed).sample 
        
        # *. Compute loss
        target = scheduler.get_velocity(err_latent, noise, t)
        if self.predict_dpt:
            loss = F.mse_loss(target_pred[:, :2][batch['msk'].repeat(1, 2, 1, 1)==1], target[:, :2][batch['msk'].repeat(1, 2, 1, 1)==1], reduction="mean")
            loss = loss * 2 / 3 + F.mse_loss(target_pred[:, 2], target[:, 2], reduction="mean") / 3
        elif self.only_dpt:
            loss = F.mse_loss(target_pred, target, reduction="mean")
        else:
            loss = F.mse_loss(target_pred[batch['msk']==1], target[batch['msk']==1], reduction="mean")
        outputs["loss"] = loss
        return outputs
   
    def forward_test(self, batch, num_inference_steps=50):
        
        # dpt, dpt_min, dpt_max = self.normalized_depth(batch['lowres_dpt'])
        # cond = torch.cat([batch['rgb'], dpt], dim=1)
        cond = batch['rgb']
        
        scheduler = self.te_scheduler
        scheduler.set_timesteps(num_inference_steps, device=cond.device)
        timesteps = scheduler.timesteps  # [T]
        Log.debug(timesteps)

        if self.predict_dpt: err_latent = torch.randn(cond[:, :3].shape, device=cond.device, dtype=cond.dtype)  
        else: err_latent = torch.randn(cond[:, :1].shape, device=cond.device, dtype=cond.dtype)  

        if self.empty_text_embed is None:
            with torch.no_grad():
                self.__encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat((cond.shape[0], 1, 1)) 

        iterable = enumerate(timesteps)
        for i, t in iterable:
            unet_input = torch.cat([cond, err_latent], dim=1)  # this order is important
            noise_pred = self.unet(unet_input, t, encoder_hidden_states=batch_empty_text_embed).sample  # [B, 4, h, w]
            err_latent = scheduler.step(noise_pred, t, err_latent).prev_sample
        torch.cuda.empty_cache()

        if self.predict_dpt: 
            return {'dpt': err_latent[:, 1:2], 'err': err_latent[:, 0:1], 'lowres_dpt': err_latent[:, 2:3]}
        elif self.only_dpt: return {'lowres_dpt': err_latent, 'err': err_latent, 'dpt': err_latent}
        else: return {'err': err_latent}
            
    
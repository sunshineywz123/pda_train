import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from hydra.utils import instantiate
from lib.utils.pylogger import Log
from tqdm.auto import tqdm
import numpy as np

EPS = 1e-3

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


class MarigoldPipeline(nn.Module, PipelineHelper):
    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215

    def __init__(self, unet_opt, vae_opt, textencoder_opt, scheduler_opt_train, scheduler_opt_test, tokenizer_opt, 
                 min_percentile=0.02,
                 max_percentile=0.98,
                 num_inference_steps=10,
                 **kwargs):
        """
        Args:
            args: pipeline
        """
        super().__init__()
        self._min_percentile = min_percentile
        self._max_percentile = max_percentile
        self._num_inference_steps = num_inference_steps
        self.tr_scheduler = DDPMScheduler(**scheduler_opt_train)
        self.te_scheduler = DDIMScheduler(**scheduler_opt_test)
        self.text_encoder = CLIPTextModel.from_pretrained(textencoder_opt.config_dir)
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_opt.config_dir)
        self.vae = AutoencoderKL.from_pretrained(vae_opt.config_dir)
        self.unet = UNet2DConditionModel.from_pretrained(unet_opt.config_dir, ignore_mismatched_sizes=True, low_cpu_mem_usage=False)
        self.empty_text_embed = None
        # ----- Freeze ----- #
        self.freeze_nets()

    def freeze_nets(self):
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)
        self.vae.eval()
        self.vae.requires_grad_(False)

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
        self.empty_text_embed = self.text_encoder(
            text_input_ids)[0].to(self.text_encoder.dtype).detach()

    def normalized_depth(self, dpt):
        B = len(dpt)
        dpt_min = torch.quantile(dpt.view(B, -1), self._min_percentile, dim=-1).view(B, 1, 1, 1)
        dpt_max = torch.quantile(dpt.view(B, -1), self._max_percentile, dim=-1).view(B, 1, 1, 1)
        if ((dpt_max - dpt_min) < EPS).any():
            dpt_max[dpt_max - dpt_min < EPS] = dpt_min[dpt_max - dpt_min < EPS] + EPS
        dpt = torch.clip(dpt, dpt_min, dpt_max)
        dpt = (dpt - dpt_min) / (dpt_max - dpt_min)
        return dpt, dpt_min, dpt_max

    def forward_train(self, batch):
        outputs = dict()
        scheduler = self.tr_scheduler

        depth_latent = self.compute_depth_latent(batch)
        cond = self.compute_cond(batch)

        # add noise to gt
        x = depth_latent
        noise = torch.randn_like(x)
        t = torch.randint(0, scheduler.config.num_train_timesteps,
                          (x.shape[0],), device=x.device).long()
        noisy_x = scheduler.add_noise(x, noise, t)
        if scheduler.config.prediction_type == 'v_prediction':
            target = scheduler.get_velocity(depth_latent, noise, t)
        elif scheduler.config.prediction_type == 'epsilon':
            target = noise

        # Encode CLIP embedding
        if self.empty_text_embed is None:
            self.__encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (cond.shape[0], 1, 1)
        )

        # *. Denoise
        unet_input = torch.cat([cond, noisy_x], dim=1)
        target_pred = self.unet(unet_input, t, encoder_hidden_states=batch_empty_text_embed).sample
        
        # *. Compute loss
        loss = F.mse_loss(target_pred, target, reduction="mean")
        outputs["loss"] = loss
        return outputs
    
    def compute_depth_latent(self, batch):
        # normalize gt and compute clean target
        dpt, dpt_min, dpt_max = self.normalized_depth(batch['depth'])
        depth_latent = self.encode_depth(dpt * 2 - 1.)
        return depth_latent
    
    def compute_cond(self, batch):
        rgb_latent = self.encode_rgb(batch['image'] * 2. - 1.)
        return rgb_latent

    def forward_test(self, batch):
        h, w = batch['image'].shape[2:]
        device = batch['image'].device
        
        scheduler = self.te_scheduler
        scheduler.set_timesteps(self._num_inference_steps, device=device)
        timesteps = scheduler.timesteps  # [T]
        
        cond = self.compute_cond(batch)
        depth_latent = torch.randn(cond[:, :4].shape, device=device, dtype=cond.dtype)

        if self.empty_text_embed is None:
            with torch.no_grad():
                self.__encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (cond.shape[0], 1, 1))

        iterable = enumerate(timesteps)
        for i, t in iterable:
            unet_input = torch.cat([cond, depth_latent], dim=1)
            noise_pred = self.unet(unet_input, t, encoder_hidden_states=batch_empty_text_embed).sample  # [B, 4, h, w]
            output = scheduler.step(noise_pred, t, depth_latent)
            depth_latent = output.prev_sample
            prediction = output.pred_original_sample
        torch.cuda.empty_cache()
        depth = self.decode_depth(prediction)
        depth = torch.clip(depth, -1.0, 1.0)
        depth = (depth + 1.0) / 2.0
        return {'depth': depth}

    def encode_depth(self, depth_in: torch.Tensor) -> torch.Tensor:
        depth_in = depth_in.repeat(1, 3, 1, 1)
        h = self.vae.encoder(depth_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        rgb_latent = mean * self.depth_latent_scale_factor
        return rgb_latent

    def decode_rgb(self, rgb_latent: torch.Tensor) -> torch.Tensor:
        rgb_latent = rgb_latent / self.rgb_latent_scale_factor
        z = self.vae.post_quant_conv(rgb_latent)
        rgb = self.vae.decoder(z)
        return rgb

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        # encode
        # rgb_in: (5, 3, H, W)
        h = self.vae.encoder(rgb_in)
        # h: (5, 8, H/8, W/8)
        moments = self.vae.quant_conv(h)
        # moments shape = h
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # 4, 4
        # scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor
        return rgb_latent

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
        depth_mean = stacked.mean(dim=1, keepdim=True)  # 三通道取均值
        return depth_mean

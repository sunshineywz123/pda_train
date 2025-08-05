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

class MarigoldPipeline(nn.Module, PipelineHelper):
    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215
    def __init__(self, args, **kwargs):
        """
        Args:
            args: pipeline
            args_clip: clip
            args_denoiser3d: denoiser3d network
        """
        super().__init__()
        self.args = args
        self.tr_scheduler = DDPMScheduler(**args.scheduler_opt_train)
        self.te_scheduler = DDIMScheduler(**args.scheduler_opt_test)
        # self.te_scheduler = DDIMScheduler.from_pretrained(args.vae_opt.config_dir.replace('vae', 'scheduler'))

        # ----- Networks ----- #
        self.text_encoder = CLIPTextModel.from_pretrained(args.textencoder_opt.config_dir)
        self.tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_opt.config_dir)
        self.vae = AutoencoderKL.from_pretrained(args.vae_opt.config_dir)
        # self.unet = UNet2DConditionModel(**args.unet_opt)
        self.unet = UNet2DConditionModel.from_pretrained(args.unet_opt.config_dir, ignore_mismatched_sizes=True, low_cpu_mem_usage=False)
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
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.text_encoder.dtype).detach()

    def forward_train(self, batch):
        outputs = dict()
        scheduler = self.tr_scheduler

        # *. Encoding
        if self.args.get('use_vae', True):
            rgb_latent = self.encode_rgb(batch['rgb'] * 2. - 1.)
            depth_latent = self.encode_depth(batch['dpt'] * 2 - 1.)
        else:
            rgb_latent = batch['rgb']
            depth_latent = batch['dpt']
            depth_latent = depth_latent.repeat(1, 3, 1, 1)
        x = depth_latent
        

        # *. Add noise
        noise = torch.randn_like(x)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (x.shape[0],), device=x.device).long()
        noisy_x = scheduler.add_noise(x, noise, t)

        # Encode CLIP embedding
        if self.empty_text_embed is None:
            self.__encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        )
        
        # *. Denoise
        unet_input = torch.cat([rgb_latent, noisy_x], dim=1)
        if self.args.get('cond_sdpt', False):
            sdpt = F.interpolate(batch['dpt'], size=noisy_x.shape[-2:], mode='nearest')
            sdpt_msk = torch.rand_like(sdpt) < (np.random.random() * 0.03 + 0.02)
            sdpt[~sdpt_msk] = 0.
            sdpt = sdpt.repeat(1, 3, 1, 1)
            unet_input = torch.cat([unet_input, sdpt], dim=1)
        target_pred = self.unet(unet_input, t, encoder_hidden_states=batch_empty_text_embed).sample 
        
        # *. Compute loss
        target = scheduler.get_velocity(x, noise, t)
        loss = F.mse_loss(target_pred, target, reduction="mean")
        if self.args.get('sdpt_msk_loss', False) and self.args.get('cond_sdpt', False):
            loss = 0.5 * loss + 0.5 * F.mse_loss(target_pred.permute(0, 2, 3, 1).reshape(-1, 4)[sdpt_msk.reshape(-1)], 
                                                 target.permute(0, 2, 3, 1).reshape(-1, 4)[sdpt_msk.reshape(-1)], reduction="mean")
        outputs["loss"] = loss
        return outputs
    
    def forward_test(self, batch, num_inference_steps=10, show_pbar=False):
        device = batch['rgb'].device
        # Set timesteps
        scheduler = self.te_scheduler
        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps  # [T]

        # rgb_latent = self.encode_rgb(batch['rgb'] * 2. - 1.)
        if self.args.get('use_vae', True):
            rgb_latent = self.encode_rgb(batch['rgb'] * 2. - 1.)
        else:
            rgb_latent = batch['rgb']
        if self.args.get('repeat_num', 1) != 1:
            rgb_latent = repeat(rgb_latent, 'b d h w -> (b r) d h w', r=self.args.repeat_num)
        
        depth_latent = torch.randn(
            rgb_latent.shape, device=device, dtype=rgb_latent.dtype
        )  

        if self.empty_text_embed is None:
            with torch.no_grad():
                self.__encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        ) 

        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)
        # depth_latent: (5, 4, 96, 72)
        # rgb_latent: (5, 4, 96, 72)
        for i, t in iterable:
            unet_input = torch.cat(
                [rgb_latent, depth_latent], dim=1
            )  # this order is important
            # 5, 8, 96, 72

            # predict the noise residual
            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_empty_text_embed
            ).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            depth_latent = scheduler.step(noise_pred, t, depth_latent).prev_sample
        torch.cuda.empty_cache()
        if self.args.get('use_vae', True):
            depth = self.decode_depth(depth_latent)
            depth = torch.clip(depth, -1.0, 1.0)
            depth = (depth + 1.0) / 2.0
        else:
            depth = depth_latent
            depth = torch.clip(depth, 0., 1.0)
        if self.args.get('repeat_num', 1) != 1:
            depth, confidence = ensemble_depths(depth[:, 0])
            return {'dpt': depth[None, None], 'confidence': confidence[None, None]}
        else:
            return {'dpt': depth}
    
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
        
        
    def decode_rgb(self, rgb_latent: torch.Tensor) -> torch.Tensor:
        rgb_latent = rgb_latent / self.rgb_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(rgb_latent)
        # z, depth_latent: (5, 4, H/8, W/8)
        rgb = self.vae.decoder(z)
        # stacked: (5, 3, H, W)
        # mean of output channels
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
        depth_mean = stacked.mean(dim=1, keepdim=True) # 三通道取均值
        return depth_mean
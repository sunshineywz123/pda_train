import torch
import importlib
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
import torch.nn.functional as F
from hydra.utils import instantiate
from einops import einsum, rearrange, repeat

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.schedulers import DDPMScheduler, DDIMScheduler
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DConditionModel, AutoencoderKL

from lib.utils.pylogger import Log
from lib.utils.depth_utils import denormalize_depth
from lib.model.marigold.marigold_utils import ensemble_depths
from lib.utils.diffusion.pipeline_helper import PipelineHelper


class MarigoldPipeline(nn.Module, PipelineHelper):
    # FIXME: What is the meaning of these two class-wise scale factors?
    rgb_latent_scale_factor = 0.18215
    dpt_latent_scale_factor = 0.18215

    def __init__(self, args, **kwargs):
        super().__init__()

        # Pipeline arguments
        self.args = args
        self.only_dpt = args.get('only_dpt', False)
        self.naive_dpt = args.get('naive_dpt', False)

        # What is this?
        if 'zerosft' in args: self.zerosft = instantiate(args.zerosft)
        self.zerosft_modulated_rgb = args.get('zerosft_modulated_rgb', False)
        self.zerosft_modulated_depth = args.get('zerosft_modulated_depth', False)

        # Diffusion schedulers
        self.tr_scheduler = DDPMScheduler(**args.scheduler_opt_train)
        self.te_scheduler = DDIMScheduler(**args.scheduler_opt_test)

        # Diffusion networks
        self.text_encoder = CLIPTextModel.from_pretrained(args.textencoder_opt.config_dir)
        self.tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_opt.config_dir)
        self.vae = AutoencoderKL.from_pretrained(args.vae_opt.config_dir)
        # Create the modified UNet model
        self.unet = instantiate(args.unet_opt)
        # Load the checkpoint
        self.unet.load_state_dict(torch.load(args.unet_opt.ckpt_path), strict=False)
        # # TODO: debug, use this as the original Marigold to see if it works
        # self.unet = UNet2DConditionModel.from_pretrained(args.unet_opt.config_dir, ignore_mismatched_sizes=False, low_cpu_mem_usage=False)

        self.empty_text_embed = None

        # Freeze networks
        self.freeze_nets()

    def freeze_nets(self):
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)
        self.vae.eval()
        self.vae.requires_grad_(False)

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
        # Initialize outputs and scheduler
        outputs = dict()
        scheduler = self.tr_scheduler

        # Step 1: Diffusion VAE encoding
        # Encode the target depth map first
        dpt_latent = self.encode_dpt(batch['dpt'] * 2. - 1.)  # (B, 4, Hz, Wz)
        # Encode all the RGB images, including the source and target images and concatenate them as the UNet condition
        rgb_inputs = torch.cat([batch['rgb'][:, None], batch['src_inps']], dim=1)  # (B, 1+S, 3, H, W)
        rgb_latent = self.encode_rgb(rearrange(rgb_inputs, 'b s c h w -> (b s) c h w') * 2. - 1.)  # (B*(1+S), 4, Hz, Wz)
        rgb_latent = rearrange(rgb_latent, '(b s) c h w -> b s c h w', s=rgb_inputs.shape[1])  # (B, 1+S, 4, Hz, Wz)
        # The target depth map latent is used as the initial depth latent
        x = dpt_latent

        # Step 2: Add noise to the latent
        noise = torch.randn_like(x[:, :4])  # (B, 4, Hz, Wz)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (x.shape[0],), device=x.device).long()  # (B,)
        x_noised = scheduler.add_noise(x, noise, t)  # (B, 4, Hz, Wz)

        # # Step 3: Prepare empty text embedding (We can skip this step since we don't use text embedding in our pipeline)
        # if self.empty_text_embed is None: self.__encode_empty_text()
        # batch_empty_text_embed = self.empty_text_embed.repeat((rgb_latent.shape[0], 1, 1))

        # Step 4: Diffusion U-Net denoising
        # Concatenate the RGB latent of the target view and noised depth latent as the input to the U-Net
        tar_rgb, tar_plc = rgb_latent[:, 0], batch['plc_embs'][:, 0]  # (B, 4, Hz, Wz), (B, 6, Hz, Wz)
        # # TODO: do experiment here to determine whether to concatenate the plucker embeddings here
        # u = torch.cat([tar_rgb, tar_plc, x_noised], dim=1)  # (B, 14, Hz, Wz)
        u = torch.cat([tar_rgb, x_noised], dim=1)  # (B, 8, Hz, Wz)
        # Prepare the conditions
        c = [rgb_latent, batch['plc_embs'], batch['epi_mask']]
        pred_noise = self.unet(u, timesteps=t, context=c)

        # Step 5: Calculate loss
        orig_noise = scheduler.get_velocity(x, noise, t)
        loss = F.mse_loss(pred_noise, orig_noise, reduction="mean")
        outputs["loss"] = loss

        return outputs
    
    def forward_test(self, batch, num_inference_steps=10):
        device = batch['rgb'].device

        # Set timesteps
        scheduler = self.te_scheduler
        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps  # (T,)

        # Step 1: Diffusion VAE encoding
        # Encode all the RGB images, including the source and target images and concatenate them as the UNet condition
        rgb_inputs = torch.cat([batch['rgb'][:, None], batch['src_inps']], dim=1)  # (B, 1+S, 3, H, W)
        rgb_latent = self.encode_rgb(rearrange(rgb_inputs, 'b s c h w -> (b s) c h w') * 2. - 1.)  # (B*(1+S), 4, Hz, Wz)
        rgb_latent = rearrange(rgb_latent, '(b s) c h w -> b s c h w', s=rgb_inputs.shape[1])  # (B, 1+S, 4, Hz, Wz)
        # Set the target depth map as random noise
        dpt_latent = torch.randn(rgb_latent[:, 0, :4].shape, device=device, dtype=rgb_latent.dtype)  # (B, 4, Hz, Wz)

        # # Step 2: Prepare empty text embedding (We can skip this step since we don't use text embedding in our pipeline)
        # if self.empty_text_embed is None:
        #     with torch.no_grad(): self.__encode_empty_text()
        # batch_empty_text_embed = self.empty_text_embed.repeat((rgb_latent.shape[0], 1, 1)) 

        # Step 3: T steps of diffusion U-Net denoising
        for i, t in enumerate(timesteps):
            # Concatenate the RGB latent of the target view and noised depth latent as the input to the U-Net
            tar_rgb, tar_plc = rgb_latent[:, 0], batch['plc_embs'][:, 0]  # (B, 4, Hz, Wz), (B, 6, Hz, Wz)
            # # TODO: do experiment here to determine whether to concatenate the plucker embeddings here
            # u = torch.cat([tar_rgb, tar_plc, dpt_latent], dim=1)  # (B, 14, Hz, Wz)
            u = torch.cat([tar_rgb, dpt_latent], dim=1)  # (B, 8, Hz, Wz)
            # Prepare the conditions
            c = [rgb_latent, batch['plc_embs'], batch['epi_mask']]
            pred_noise = self.unet(u, timesteps=t[None].expand(dpt_latent.shape[0]), context=c)  # (B, 4, Hz, Wz)
            dpt_latent = scheduler.step(pred_noise, t, dpt_latent).prev_sample

        # # TODO: debug
        # # Step 2: Prepare empty text embedding (We can skip this step since we don't use text embedding in our pipeline)
        # if self.empty_text_embed is None:
        #     with torch.no_grad(): self.__encode_empty_text()
        # batch_empty_text_embed = self.empty_text_embed.repeat((rgb_latent.shape[0], 1, 1)) 
        # # Step 3: T steps of diffusion U-Net denoising
        # for i, t in enumerate(timesteps):
        #     # Concatenate the RGB latent of the target view and noised depth latent as the input to the U-Net
        #     u = torch.cat([rgb_latent[:, 0], dpt_latent], dim=1)  # (B, 8, Hz, Wz)
        #     pred_noise = self.unet(u, t, encoder_hidden_states=batch_empty_text_embed).sample  # (B, 4, Hz, Wz)
        #     dpt_latent = scheduler.step(pred_noise, t, dpt_latent).prev_sample

        # Step 4: Get the raw depth map and perform postprocess
        torch.cuda.empty_cache()
        depth = self.decode_dpt(dpt_latent)
        depth = torch.clip(depth, -1.0, 1.0)
        depth = (depth + 1.0) / 2.0

        return {'dpt': depth}

    def encode_dpt(self, dpt_in: torch.Tensor) -> torch.Tensor:
        """
        Encode depth image into latent.

        Args:
            dpt_in (`torch.Tensor`), (B, 1, H, W): Input depth image to be encoded.

        Returns:
            `torch.Tensor` (B, 4, Hz, Wz): Depth latent.
        """
        # Repeat depth map to 3 channels to match the input size of the VAE encoder
        dpt_in = dpt_in.repeat(1, 3, 1, 1)  # (B, 3, H, W)

        # Encode depth map
        h = self.vae.encoder(dpt_in)  # (B, 8, H/8, W/8)

        # Quantize depth map
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)  # (B, 4, H/8, W/8)

        # Scale latent
        dpt_latent = mean * self.dpt_latent_scale_factor

        return dpt_latent

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`), (B, 3, H, W): Input RGB image to be encoded.

        Returns:
            `torch.Tensor` (B, 4, Hz, Wz): Image latent.
        """
        # Encode image to latent space
        h = self.vae.encoder(rgb_in)  # (B, 8, Hz, Wz)

        # Quantize depth map
        moments = self.vae.quant_conv(h)  # (B, 8, Hz, Wz)
        mean, logvar = torch.chunk(moments, 2, dim=1)

        # Scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor

        return rgb_latent

    def decode_dpt(self, dpt_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (`torch.Tensor`): Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        # Scale latent
        dpt_latent = dpt_latent / self.dpt_latent_scale_factor

        # VAE decode to the original space
        z = self.vae.post_quant_conv(dpt_latent)  # (B, 4, Hz, Wz)
        stacked = self.vae.decoder(z)  # (B, 3, H, W)

        # Mean of output channels
        depth_mean = stacked.mean(dim=1, keepdim=True)  # mean of the 3 channels

        return depth_mean

    def decode_rgb(self, rgb_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (`torch.Tensor`): Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        # Scale latent
        rgb_latent = rgb_latent / self.rgb_latent_scale_factor

        # VAE decode to the original space
        z = self.vae.post_quant_conv(rgb_latent)  # (B, 4, Hz, Wz)
        rgb = self.vae.decoder(z)  # (B, 3, H, W)

        return rgb

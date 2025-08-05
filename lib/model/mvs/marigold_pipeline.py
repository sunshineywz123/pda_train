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
from lib.utils.math_utils import affine_padding, affine_inverse
from lib.utils.simplerecon.cost_volume import CostVolumeManager, FastFeatureVolumeManager


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

        # Trimmed ControlNet by SUPIR, see http://arxiv.org/abs/2401.13627
        if 'zerosft' in args: self.zerosft = instantiate(args.zerosft)
        self.zerosft_modulated_rgb = args.get('zerosft_modulated_rgb', False)
        self.zerosft_modulated_mvs = args.get('zerosft_modulated_mvs', False)

        # Diffusion schedulers
        self.tr_scheduler = DDPMScheduler(**args.scheduler_opt_train)
        self.te_scheduler = DDIMScheduler(**args.scheduler_opt_test)

        # Diffusion networks
        self.text_encoder = CLIPTextModel.from_pretrained(args.textencoder_opt.config_dir)
        self.tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_opt.config_dir)
        self.vae = AutoencoderKL.from_pretrained(args.vae_opt.config_dir)
        self.unet = UNet2DConditionModel.from_pretrained(args.unet_opt.config_dir, ignore_mismatched_sizes=True, low_cpu_mem_usage=False)

        # Text condition encoder
        self.empty_text_embed = None

        # Multi-view condition encoder
        if 'cost_volume' in args: self.cost_volume = CostVolumeManager(**args.cost_volume)
        if 'fast_feature_volume' in args: self.cost_volume = FastFeatureVolumeManager(**args.fast_feature_volume)

        # Freeze networks
        self.freeze_nets()

    def freeze_nets(self):
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)
        self.vae.eval()
        self.vae.requires_grad_(False)

    def __encode_empty_text(self):
        """ Encode text embedding for empty prompt
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

    def compute_conditions(self, batch):
        # Encode all the RGB images, including the source and target images and concatenate them as the UNet condition
        rgb_inputs = torch.cat([batch['rgb'][:, None], batch['src_inps']], dim=1)  # (B, 1+S, 3, H, W)
        rgb_latent = self.encode_rgb(rearrange(rgb_inputs, 'b s c h w -> (b s) c h w') * 2. - 1.)  # (B*(1+S), 4, Hz, Wz)
        rgb_latent = rearrange(rgb_latent, '(b s) c h w -> b s c h w', s=rgb_inputs.shape[1])  # (B, 1+S, 4, Hz, Wz)
        # Split the target latent features and source latent features
        tar_latent = rgb_latent[:,  0]  # (B, 4, Hz, Wz)
        src_latent = rgb_latent[:, 1:]  # (B, S, 4, Hz, Wz)

        # Prepare target and source extrinsic and intrinsic matrices
        src_exts = affine_padding(batch['src_exts']) @ affine_inverse(batch['tar_ext'])[:, None]  # (B, S, 4, 4)
        tar_exts = affine_padding(batch['tar_ext'])[:, None] @ affine_inverse(batch['src_exts'])  # (B, S, 4, 4)
        src_ixts = affine_padding(batch['src_ixts'])  # (B, S, 4, 4)
        tar_iixt = torch.inverse(affine_padding(batch['tar_ixt']))  # (B, 4, 4)

        # Build the cost volume
        # (B, D, Hz, Wz), (B, Hz, Wz), ..., (B, Hz, Wz)
        cost_volume, lowest_cost, _, overall_mask_bhw = self.cost_volume(cur_feats=tar_latent,
                                                                         src_feats=src_latent,
                                                                         src_extrinsics=src_exts,
                                                                         src_poses=tar_exts,
                                                                         src_Ks=src_ixts,
                                                                         cur_invK=tar_iixt,
                                                                         min_depth=batch['n'][..., None, None, None],
                                                                         max_depth=batch['f'][..., None, None, None],
                                                                         return_mask=True)

        # Prepare the conditions
        if self.zerosft_modulated_rgb:
            conditions = self.zerosft(cost_volume, tar_latent)  # (B, 4, Hz, Wz)
        else:
            conditions = torch.cat([tar_latent, cost_volume], dim=1)  # (B, 4+D, Hz, Wz)

        return conditions

    def forward_train(self, batch):
        # Initialize outputs and scheduler
        outputs = dict()
        scheduler = self.tr_scheduler

        # Step 1: Diffusion VAE encoding
        # Encode the target depth map first
        dpt_latent = self.encode_dpt(batch['dpt'] * 2. - 1.)  # (B, 4, Hz, Wz)
        # Compute the conditions
        conditions = self.compute_conditions(batch)

        # Step 2: Add noise to the latent
        # The target depth map latent is used as the initial depth latent
        x = dpt_latent
        noise = torch.randn_like(x[:, :4])  # (B, 4, Hz, Wz)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (x.shape[0],), device=x.device).long()  # (B)
        x_noised = scheduler.add_noise(x, noise, t)  # (B, 4, Hz, Wz)

        # Step 3: Prepare empty text embedding (We can skip this step since we don't use text embedding in our pipeline)
        if self.empty_text_embed is None: self.__encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat((dpt_latent.shape[0], 1, 1))

        # Step 4: Diffusion U-Net denoising
        # Concatenate the conditions and noised depth latent as the input to the U-Net
        unet_input = torch.cat([conditions, x_noised], dim=1)  # (B, C, Hz, Wz)
        pred_noise = self.unet(unet_input, t, encoder_hidden_states=batch_empty_text_embed).sample 

        # Step 5: Calculate loss
        orig_noise = scheduler.get_velocity(x, noise, t)
        loss = F.mse_loss(pred_noise, orig_noise, reduction="mean")
        outputs["loss"] = loss

        return outputs
    
    def forward_test(self, batch, num_inference_steps=10):
        # Get device
        device = batch['rgb'].device

        # Set timesteps
        scheduler = self.te_scheduler
        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps  # (T,)

        # Step 1: Diffusion VAE encoding
        # Compute the conditions
        conditions = self.compute_conditions(batch)

        # Step 2: Create the target depth latent as random noise
        dpt_latent = torch.randn(conditions[:, :4].shape, device=device, dtype=conditions.dtype)  # (B, 4, Hz, Wz)

        # Step 3: Prepare empty text embedding (We can skip this step since we don't use text embedding in our pipeline)
        if self.empty_text_embed is None:
            with torch.no_grad(): self.__encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat((dpt_latent.shape[0], 1, 1)) 

        # Step 4: T steps of diffusion U-Net denoising
        for i, t in enumerate(timesteps):
            # Concatenate the conditions and noised depth latent as the input to the U-Net
            unet_input = torch.cat([conditions, dpt_latent], dim=1)  # (B, C, Hz, Wz)
            pred_noise = self.unet(unet_input, t, encoder_hidden_states=batch_empty_text_embed).sample  # (B, 4, Hz, Wz)
            dpt_latent = scheduler.step(pred_noise, t, dpt_latent).prev_sample  # (B, 4, Hz, Wz)

        # Step 5: Get the raw depth map and perform postprocess
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

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
    AutoencoderKL,
)
from transformers import CLIPTextModel, CLIPTokenizer
from lib.utils.diffusion.pipeline_helper import PipelineHelper
from .unet import UNet2DConditionModel
# from lib.model.supermotion.utils.motion3d_endecoder import EnDecoderBase

from lib.utils.diffusion.utils import randlike_shape
# from lib.utils.check_utils import check_equal_get_one
from lib.utils.geo.triangulation import triangulate_2d_3d
from lib.model.marigold.marigold_utils import ensemble_depths

class NVDPTPipeline(nn.Module, PipelineHelper):
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
        
        
        self.last_noise = None

        # ----- Freeze ----- #
        self.freeze_nets()

    def freeze_nets(self):
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)
        self.vae.eval()
        self.vae.requires_grad_(False)
        for parameter in self.unet.named_parameters():
            if 'motion_module' not in parameter[0]:
                parameter[1].requires_grad_(False)
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
        B, N_V = batch['rgbs'].shape[:2]
        rgbs = batch['rgbs'].reshape(-1, *batch['rgbs'].shape[2:])
        dpts = batch['dpts'].reshape(-1, *batch['dpts'].shape[2:])
        rgb_latent = self.encode_rgb(rgbs * 2. - 1.)
        depth_latent = self.encode_depth(dpts * 2 - 1.)
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
        target_pred = self.unet(unet_input, t, encoder_hidden_states=batch_empty_text_embed,batch_size=B).sample 
        
        # *. Compute loss
        target = scheduler.get_velocity(x, noise, t)
        loss = F.mse_loss(target_pred, target, reduction="mean")
        if self.args.get('sdpt_msk_loss', False) and self.args.get('cond_sdpt', False):
            loss = 0.5 * loss + 0.5 * F.mse_loss(target_pred.permute(0, 2, 3, 1).reshape(-1, 4)[sdpt_msk.reshape(-1)], 
                                                 target.permute(0, 2, 3, 1).reshape(-1, 4)[sdpt_msk.reshape(-1)], reduction="mean")
        outputs["loss"] = loss
        return outputs
    
    def forward_test(self, batch, num_inference_steps=10, show_pbar=False):
        device = batch['rgbs'].device
        # Set timesteps
        scheduler = self.te_scheduler
        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps  # [T]

        # rgb_latent = self.encode_rgb(batch['rgb'] * 2. - 1.)
        B, N_V = batch['rgbs'].shape[:2]
        rgbs = batch['rgbs'].reshape(-1, *batch['rgbs'].shape[2:])
        rgb_latent = self.encode_rgb(rgbs * 2. - 1.)
        # TODO: 
        if self.args.get('repeat_num', 1) != 1:
            rgb_latent = repeat(rgb_latent, 'b d h w -> (b r) d h w', r=self.args.repeat_num)

        depth_latent = torch.randn(
            rgb_latent.shape, device=device, dtype=rgb_latent.dtype
        )
        # video_length = len(depth_latent)
        # overlap = video_length // 2 + 1
        # if self.last_noise is not None:
        #     depth_latent[:overlap] = self.last_noise[-overlap:]
        # self.last_noise = depth_latent.clone()
          
        # TODO: 保存噪音，控制噪音一样

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
                unet_input, t, encoder_hidden_states=batch_empty_text_embed, batch_size=B
            ).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            depth_latent = scheduler.step(noise_pred, t, depth_latent).prev_sample
        torch.cuda.empty_cache()
        depth = self.decode_depth(depth_latent)

        # clip prediction
        depth = torch.clip(depth, -1.0, 1.0)
        # shift to [0, 1]
        depth = (depth + 1.0) / 2.0
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

    def triangulate_x0(self, x0_3d_, inputs):
        """
        Args:
            x0_3d_: tensor that will be handled by self.decode and self.encode, shape example (B, D, L)
        """
        # # DEBUG: as generation
        # x_triag = self.decoder_motion3d(x0_3d_)  # (B, L, 22, 3)
        # return x0_3d_, x_triag

        # obs view
        proj_mode = "persp"
        c_p2d_cv = rearrange(inputs["c_p2d_cv"], "b l j c -> b (l j) c")
        T_ayfz2c = inputs["T_ayfz2c"]  # (B, 4, 4)
        # weight_2d = inputs["t3d"] / self.num_inference_steps  # Linear annealing (not-working)
        # weight_2d = weight_2d ** (0.25)  # Smooth process (not-working)
        weight_2d = 1.0
        # if inputs["t3d"] / self.num_inference_steps < 0.2: # (not-working)
        #     weight_2d = 0.0

        # 3d motion prior
        pred_ayf_motion = self.decoder_motion3d(x0_3d_)  # (B, L, 22, 3)
        B, L, J, _ = pred_ayf_motion.shape

        # Do constraint on pred_ayf_motion
        pred_ayf_motion_ = rearrange(pred_ayf_motion, "b l j c -> b 1 (l j) c")
        if "init_motion_ayfz" in inputs:  # Use init Joint3d as another regularization
            init_motion_ayfz = rearrange(inputs["init_motion_ayfz"], "b l j c -> b 1 (l j) c")
            pred_ayf_motion_ = torch.cat([pred_ayf_motion_, init_motion_ayfz], dim=1)

        pred_ayf_motion_ = triangulate_2d_3d(
            T_ayfz2c[:, None], c_p2d_cv[:, None], proj_mode, pred_ayf_motion_, weight_2d=weight_2d
        )
        pred_ayf_motion_ = rearrange(pred_ayf_motion_, "b (l j) c -> b l j c", l=L, j=J)
        x_triag = pred_ayf_motion_.clone()
        x0_inpaint = self.encoder_motion3d(pred_ayf_motion_)  # (B, D, L)

        return x0_inpaint, x_triag


def randomly_set_null_condition(text, img_seq_fid):
    """
    Args:
        text: List of str
        img_seq_fid: (B, I), -1 indicates padding
    """
    # To support classifier-free guidance, randomly set-to-unconditioned
    B = len(text)
    uncond_prob = 0.1

    # text
    text_mask = torch.rand(B) < uncond_prob
    text_ = ["" if m else t for m, t in zip(text_mask, text)]

    # img_seq_fid
    img_seq_fid_mask = torch.rand(B, img_seq_fid.shape[1]) < uncond_prob
    img_seq_fid_ = img_seq_fid.clone()
    img_seq_fid_[img_seq_fid_mask] = -1

    # Follow Zero123, we randomly drop all conditions # TODO: the probability is not clear @phj
    allcond_mask = torch.rand(B) < uncond_prob
    text_ = ["" if m else t for m, t in zip(allcond_mask, text_)]
    img_seq_fid_[allcond_mask] = -1

    return text_, img_seq_fid_

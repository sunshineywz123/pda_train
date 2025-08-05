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
        """
        super().__init__()
        self.args = args
        self.upsample = args.get('upsample', False) # 上采样设置
        self.only_dpt = args.get('only_dpt', False) # 仅使用dpt作为condition
        self.naive_dpt = args.get('naive_dpt', False) # 使用low-res dpt作为condition
        self.linear_sample = args.get('linear_sample', False) # 如何上采样lowres_dpt
        self.down_linear_sample = args.get('down_linear_sample', False) # 如何下采样lowres_dpt
        
        # 如何simulate lowres dpt
        # 默认使用nearest downsample, -> linear
        # 比例方式
        # 1. 默认nearest降采样1/8
        # 2. nearest降采样1/8 - 192-256；0.3, 0.4, 0.3
        self.range_ratio = args.get('range_ratio', [1., 0., 0.])
        self.simulate_linear_sample = args.get('simulate_linear_sample', False) # 简化下采样
        self.inference_method = args.get('inference_method', 0) # 默认inference method
        
        self.simp_bilinear = args.get('simp_bilinear', False) # 直接进行上采样
        if self.simp_bilinear:
            return
        if 'zerosft' in args:
            self.zerosft = instantiate(args.zerosft)
        self.zerosft_modulated_rgb = args.get('zerosft_modulated_rgb', False)
        self.zerosft_modulated_depth = args.get('zerosft_modulated_depth', False)
        
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
        
    def normalized_depth(self, dpt):
        dpt_min, dpt_max = torch.quantile(dpt, self.args.get('min_percentile', 0.02)), torch.quantile(dpt, self.args.get('max_percentile', 0.98))
        dpt_max = (dpt_min + 0.01) if (dpt_max - dpt_min < 1e-6) else dpt_max
        dpt = torch.clip(dpt, dpt_min, dpt_max)
        dpt = (dpt - dpt_min) / (dpt_max - dpt_min)
        return dpt, dpt_min, dpt_max
    
    def get_simdpt(self, dpt, h, w, inference_method=-1):
        """
        dpt: (B, 1, H, W)
        h: int, height of rgb height / 8
        w: int, width of rgb width / 8
        """
        if inference_method == -1: # training mode
            simulate_ratio_method = np.random.choice([0, 1, 2], p=self.range_ratio)
        else:
            simulate_ratio_method = inference_method
        if simulate_ratio_method == 0: tar_h, tar_w = h, w
        elif simulate_ratio_method == 1: tar_h, tar_w = 192, 256
        elif simulate_ratio_method == 2:
            # tar_h, tar_w均匀从 (h, w) 到 (192, 256), 均为整数，h:w=3:4
            # 找到h到192之间可以被3整除的数字，随机采样
            # tar_h = np.random.choice([i for i in range(h, 192) if i % 3 == 0])
            start, end = h if h % 3 == 0 else h + (3 - h % 3), 192
            tar_h = np.random.choice(np.arange(start, end, 3))
            tar_w = int(tar_h * 4 / 3)
        else:
            raise ValueError(f"Invalid inference method: {inference_method}")
        dpt_h = dpt.shape[2] 
        tar_h, tar_w = int(tar_h), int(tar_w)
        if dpt_h != tar_h: 
            if self.simulate_linear_sample: dpt = F.interpolate(dpt, (tar_h, tar_w), mode='bilinear', align_corners=False)
            else: dpt = F.interpolate(dpt, (tar_h, tar_w), mode='nearest')
        return dpt
            
    def compute_cond(self, rgb_latent, normalized_dpt, dpt_min, dpt_max, batch, inference=False):
        if not self.upsample: return rgb_latent
        import ipdb; ipdb.set_trace()
        
        h_8, w_8 = rgb_latent.shape[2:]
        h, w = h_8*8, w_8*8
        if inference: lowres_dpt = normalized_dpt
        else: lowres_dpt = self.get_simdpt(normalized_dpt, h_8, w_8)
        h_low, w_low = lowres_dpt.shape[2:]
        # upsample lowres dpt to rgb shape
        if self.linear_sample: dpt_up = F.interpolate(lowres_dpt, (h, w), mode='bilinear', align_corners=False)
        else: dpt_up = F.interpolate(lowres_dpt, (h, w), mode='nearest')
        dpt_up_latent = self.encode_depth(dpt_up * 2 - 1.)
        
        # downsample lowres dpt to rgb/8 shape
        if h_low != h_8: 
            if self.down_linear_sample: dpt_down = F.interpolate(lowres_dpt, (h_8, w_8), mode='bilinear', align_corners=False)
            else: dpt_down = F.interpolate(lowres_dpt, (h_8, w_8), mode='nearest')
        else: dpt_down = lowres_dpt
        
        # compute condtion 
        if self.only_dpt: cond = torch.cat([dpt_up_latent, dpt_down], dim=1)
        elif self.zerosft_modulated_rgb: cond = self.zerosft(torch.cat([dpt_up_latent, dpt_down], dim=1), rgb_latent)
        elif self.zerosft_modulated_depth: cond = self.zerosft(rgb_latent, torch.cat([dpt_up_latent, dpt_down], dim=1))
        elif self.naive_dpt: cond = torch.cat([rgb_latent, dpt_down], dim=1)
        else: cond = torch.cat([rgb_latent, dpt_up_latent, dpt_down], dim=1)
        
        return cond
        
        

    def forward_train(self, batch):
        outputs = dict()
        scheduler = self.tr_scheduler
        
        # normalize gt and compute clean target
        # augmentation
        dpt, dpt_min, dpt_max = self.normalized_depth(batch['dpt'])
        depth_latent = self.encode_depth(dpt * 2 - 1.)
        
        # compute condition 
        # cond需要用低分辨率depth, normalize参数, 针对depth upsampling任务
        rgb_latent = self.encode_rgb(batch['rgb'] * 2. - 1.)
        cond = self.compute_cond(rgb_latent, dpt, dpt_min, dpt_max, batch)
        
        # add noise to gt
        x = depth_latent
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
        unet_input = torch.cat([cond, noisy_x], dim=1)
        target_pred = self.unet(unet_input, t, encoder_hidden_states=batch_empty_text_embed).sample 
        
        # *. Compute loss
        target = scheduler.get_velocity(x, noise, t)
        loss = F.mse_loss(target_pred, target, reduction="mean")
        outputs["loss"] = loss
        return outputs
   
    def forward_test(self, batch, num_inference_steps=10):
        h, w = batch['rgb'].shape[2:]
        h_8, w_8 = h//8, w//8
        
        if 'lowres_dpt' in batch: lowres_dpt = batch['lowres_dpt']
        else: lowres_dpt = batch['dpt']
        import ipdb; ipdb.set_trace()
        lowres_dpt = self.get_simdpt(lowres_dpt, h_8, w_8, inference_method=self.inference_method)
        lowres_dpt, dpt_min, dpt_max = self.normalized_depth(lowres_dpt)
        
        # dpt, dpt_min, dpt_max = self.normalized_depth(batch['dpt'])
        # dpt_latent = self.encode_depth(dpt * 2 - 1.)
        # dpt = self.decode_depth(dpt_latent)
        # dpt = torch.clip(dpt, -1.0, 1.0)
        # dpt = (dpt + 1.0) / 2.0
        # dpt = dpt * (dpt_max - dpt_min) + dpt_min
        # return {'dpt': dpt}
        
        if self.simp_bilinear: return {'dpt': F.interpolate(lowres_dpt, (h, w), mode='bilinear', align_corners=False) * (dpt_max - dpt_min) + dpt_min}

        device = batch['rgb'].device
        # Set timesteps
        scheduler = self.te_scheduler
        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps  # [T]

        rgb_latent = self.encode_rgb(batch['rgb'] * 2. - 1.)
        cond = self.compute_cond(rgb_latent, lowres_dpt, dpt_min, dpt_max, batch, inference=True)
        
        if self.args.get('repeat_num', 1) != 1:
            raise NotImplementedError
            rgb_latent = repeat(rgb_latent, 'b d h w -> (b r) d h w', r=self.args.repeat_num)
        
        depth_latent = torch.randn(rgb_latent[:, :4].shape, device=device, dtype=rgb_latent.dtype)  

        if self.empty_text_embed is None:
            with torch.no_grad():
                self.__encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat((rgb_latent.shape[0], 1, 1)) 

        iterable = enumerate(timesteps)
        for i, t in iterable:
            unet_input = torch.cat([cond, depth_latent], dim=1)  # this order is important
            noise_pred = self.unet(unet_input, t, encoder_hidden_states=batch_empty_text_embed).sample  # [B, 4, h, w]
            depth_latent = scheduler.step(noise_pred, t, depth_latent).prev_sample
        torch.cuda.empty_cache()
        depth = self.decode_depth(depth_latent)
        depth = torch.clip(depth, -1.0, 1.0)
        depth = (depth + 1.0) / 2.0
        
        if self.upsample: depth = depth * (dpt_max - dpt_min) + dpt_min
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
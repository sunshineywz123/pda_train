from typing import Any, Dict
import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
from lib.utils.pylogger import Log
from os.path import join
import os
import imageio
import numpy as np
import cv2
# import psnr metric from scikit
from skimage.metrics import peak_signal_noise_ratio as psnr

from lib.utils.dpt.eval_utils import recover_metric_depth, evaluate_rel_err
from einops import rearrange

class StereoModel(pl.LightningModule):
    def __init__(
        self,
        pipeline,
        optimizer=None,
        scheduler_cfg=None,
        args=None,
        ignored_weights_prefix=["pipeline.text_encoder",
                                "pipeline.vae"],
        **kwargs,
    ):
        super().__init__()
        self.pipeline = instantiate(pipeline, _recursive_=False)
        self.optimizer = instantiate(optimizer)
        self.scheduler_cfg = scheduler_cfg

        # Options
        self.args = args
        self.seed = args.seed
        self.rng_type = args.rng_type  # [const, random, seed_plus_idx]
        self.ignored_weights_prefix = ignored_weights_prefix
        self.output_dir = join(args.get('output_dir'), args.get('save_tag'))

        # The test step is the same as validation
        self.idx = 0
        self.test_step = self.validation_step

    def training_step(self, batch, batch_idx):
        # forward and compute loss
        B = self.trainer.train_dataloader.batch_size
        outputs = self.pipeline.forward_train(batch)
        loss = outputs["loss"]
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=B)
        return outputs
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None) -> None:
        """Task-specific predict step. CAP or GEN."""
        outputs = self.pipeline.forward_test(batch)
        self.save_imgs(batch, outputs, batch_idx)

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        """Task-specific validation step. CAP or GEN."""
        outputs = self.pipeline.forward_test(batch)
        self.save_imgs(batch, outputs, batch_idx)
        right_img = batch['right_img'].permute(0, 2, 3, 1).detach().cpu().numpy()
        pred_img = outputs.permute(0, 2, 3, 1).detach().cpu().numpy()
        for i in range(len(right_img)):
            psnr_val = psnr(right_img[i], pred_img[i])
            self.log("val/psnr", psnr_val, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
    def compute_metrics(self, batch, outputs):
        depth_type = batch['meta']['depth_norm'][0]
        gt_dpt = batch['dpt'][:, 0].detach().cpu().numpy()
        pred_dpt = outputs['dpt'][:, 0].detach().cpu().numpy()
        if depth_type == 'simp':
            import ipdb; ipdb.set_trace()
        elif depth_type == 'disp':
            gt_dpt = 1. / np.clip(gt_dpt, 1/80., None)
            pred_dpt = 1. / np.clip(pred_dpt, 1/80., None)
        elif depth_type == 'log':
            gt_dpt = np.exp(gt_dpt * (np.log(80.) - np.log(0.5)) + np.log(0.5))
            pred_dpt = np.exp(pred_dpt * (np.log(80.) - np.log(0.5)) + np.log(0.5))
        elif depth_type == 'nonorm':
            pass
        else:
            import ipdb; ipdb.set_trace()
            
        mask_invalid = np.zeros_like(gt_dpt[0]).astype(np.uint8)
        # range crop 
        # mask_invalid[gt_dpt < 0.5] = 1
        # mask_invalid[gt_dpt > 10.] = 1
        # eigen crop
        mask_invalid[:45,] = 1
        mask_invalid[:, :41] = 1
        mask_invalid[471:] = 1
        mask_invalid[:, 601:] = 1
        mask_invalid = mask_invalid.astype(np.bool_)
        
        metrcs_return = {}
        for b in range(len(gt_dpt)):
            metric_depth = recover_metric_depth(pred_dpt[b], gt_dpt[b], mask0=~mask_invalid)
            metrics_dict = evaluate_rel_err(metric_depth, gt_dpt[b], mask_invalid)
            for k, v in metrics_dict.items():
                if k not in metrcs_return:
                    metrcs_return[k] = []
                metrcs_return[k].append(v)
        for k, v in metrcs_return.items():
            metrcs_return[k] = np.mean(v)
        return metrcs_return
        
        # rmse = np.sqrt(np.mean((gt_dpt - pred_dpt)**2))
        # gt_dpt = np.clip(gt_dpt, 1e-6, None)
        # rel = np.mean(np.abs(gt_dpt - pred_dpt) / gt_dpt)
        # return {'rmse': rmse, 'rel': rel}
        
    def configure_optimizers(self):
        def get_lr(lr_table, param_k, default_lr):
            lr = None
            g = 'default'
            for k, v in lr_table.items():
                if k in param_k:
                    if lr is None: 
                        lr = v
                        g = k
                    else:
                        # error
                        import ipdb; ipdb.set_trace()
            return g, lr if lr is not None else default_lr
        group_table = {}
        params = []
        for k, v in self.pipeline.named_parameters():
            if v.requires_grad:
                group, lr = get_lr(self.args.lr_table, k, 1e-5)
                if group not in group_table:
                    group_table[group] = len(group_table)
                    params.append({'params': [v], 'lr': lr, 'name': group})
                else:
                    params[group_table[group]]['params'].append(v)
        optimizer = self.optimizer(params=params)
        if self.scheduler_cfg is None:
            return optimizer
        scheduler_cfg = self.scheduler_cfg
        scheduler_cfg["scheduler"] = instantiate(scheduler_cfg["scheduler"], optimizer=optimizer)
        return [optimizer], [scheduler_cfg]

    # ============== Utils ================= #

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        for ig_keys in self.ignored_weights_prefix:
            Log.debug(f"Remove key `{ig_keys}' from checkpoint.")
            for k in list(checkpoint["state_dict"].keys()):
                if k.startswith(ig_keys):
                    checkpoint["state_dict"].pop(k)
        super().on_save_checkpoint(checkpoint)

    def load_pretrained_model(self, ckpt_path, ckpt_type):
        """Load pretrained checkpoint, and assign each weight to the corresponding part."""
        Log.info(f"Loading ckpt type `{ckpt_type}': {ckpt_path}")
        state_dict = torch.load(ckpt_path, "cpu")["state_dict"]
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        real_missing = []
        for k in missing:
            miss = True
            for ig_keys in self.ignored_weights_prefix:
                if k.startswith(ig_keys):
                    miss = False
            if miss:
                real_missing.append(k)
        if len(real_missing) > 0:
            Log.warn(f"Missing keys: {real_missing}")
        if len(unexpected) > 0:
            Log.error(f"Unexpected keys: {unexpected}")
            
            
    def save_img(self, pred, rgb, name):
        from lib.utils.vis_utils import colorize_depth_maps
        depth_min, depth_max = pred.min(), pred.max()
        depth_norm = (pred - depth_min) / (depth_max - depth_min)
        depth_vis = colorize_depth_maps(depth_norm, 0., 1.)[0].transpose((1, 2, 0))
        img_set = {'depth_vis': depth_vis, 'rgb': rgb}
        img_path = join(self.output_dir, '{}'.format(name))
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        imageio.imwrite(img_path, (np.concatenate([img_set[k] for k in img_set], axis=1) * 255.).astype(np.uint8))
            
    def save_imgs(self, batch, output, batch_idx):
        left_img = batch['left_img'].permute(0, 2, 3, 1).detach().cpu().numpy()
        right_img = batch['right_img'].permute(0, 2, 3, 1).detach().cpu().numpy()
        warped_img = batch['warped_img'].permute(0, 2, 3, 1).detach().cpu().numpy()
        pred_img = output.permute(0, 2, 3, 1).detach().cpu().numpy()
        
        for i in range(len(left_img)):
            save_img = np.concatenate([left_img[i], right_img[i]], axis=1)
            save_img = np.concatenate([save_img, np.concatenate([warped_img[i], pred_img[i]], axis=1)], axis=0)
            save_path = join(self.output_dir, '{:04d}_{:04d}.png'.format(batch_idx, i))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            imageio.imwrite(save_path, (save_img * 255).astype(np.uint8))
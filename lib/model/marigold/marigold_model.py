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

from lib.utils.check_utils import check_equal_get_one
from lib.utils.dpt.eval_utils import recover_metric_depth, evaluate_rel_err
# from lib.utils.geo_transform import T_transforms_points, project_p2d
from einops import rearrange

class MarigoldModel(pl.LightningModule):
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
        self.seed = args.seed
        self.rng_type = args.rng_type  # [const, random, seed_plus_idx]
        self.ignored_weights_prefix = ignored_weights_prefix
        self.output_dir = join(args.get('output_dir'), args.get('save_tag'))

        # The test step is the same as validation
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
        self.save_imgs(batch, outputs)

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        """Task-specific validation step. CAP or GEN."""
        outputs = self.pipeline.forward_test(batch)
        metrics_dict = self.compute_metrics(batch, outputs)
        B = len(batch['rgb'])
        # B = self.trainer.val_dataloaders[dataloader_idx].batch_size
        for k, v in metrics_dict.items():
            self.log(f"val/{k}", v, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=B)
        # Log.info(batch['meta']['rgb_name'][0], metrics_dict)
    
    def compute_metrics(self, batch, outputs):
        depth_type = batch['meta']['depth_norm'][0]
        gt_dpt = batch['dpt'][0, 0].detach().cpu().numpy()
        pred_dpt = outputs['dpt'][0, 0].detach().cpu().numpy()
        if depth_type == 'simp':
            pass
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
        mask_invalid = gt_dpt < 1e-8
        # mask_invalid[45:471, 41:601] = 1
        mask_invalid[:45,] = 1
        mask_invalid[:, :41] = 1
        mask_invalid[471:] = 1
        mask_invalid[:, 601:] = 1
        mask_invalid = mask_invalid.astype(np.bool_)
            
        metric_depth = recover_metric_depth(pred_dpt, gt_dpt, ~mask_invalid)
        metrics_dict = evaluate_rel_err(metric_depth, gt_dpt, mask_invalid)
        return metrics_dict
        
        # rmse = np.sqrt(np.mean((gt_dpt - pred_dpt)**2))
        # gt_dpt = np.clip(gt_dpt, 1e-6, None)
        # rel = np.mean(np.abs(gt_dpt - pred_dpt) / gt_dpt)
        # return {'rmse': rmse, 'rel': rel}

    def configure_optimizers(self):
        params = []
        for k, v in self.pipeline.named_parameters():
            if v.requires_grad:
                params.append(v)
        optimizer = self.optimizer(params=params)

        if self.scheduler_cfg is None:
            return optimizer

        scheduler_cfg = self.scheduler_cfg
        scheduler_cfg["scheduler"] = instantiate(scheduler_cfg["scheduler"], optimizer=optimizer)

        return [optimizer], [scheduler_cfg]

    # ============== Utils ================= #

    def get_generatror(self, batch_idx):
        """Fix the random seed for each batch at sampling stage."""
        generator = torch.Generator(self.device)
        if self.rng_type == "const":
            generator.manual_seed(self.seed)
        elif self.rng_type == "random":
            pass
        elif self.rng_type == "seed_plus_idx":
            generator.manual_seed(self.seed + batch_idx)
        else:
            raise ValueError(f"rng_type `{self.rng_type}' is not supported.")
        generator.manual_seed(self.seed)
        return generator

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
            
    def save_imgs(self, batch, output):
        from lib.utils.vis_utils import colorize_depth_maps
        rgb = batch['rgb'][0].permute(1, 2, 0).detach().cpu().numpy()
        depth = output['dpt'][0, 0].detach().cpu().numpy()
        depth_min, depth_max = depth.min(), depth.max()
        if depth_min != 0. or depth_max != 1.:
            Log.info('Non 0-1, depth_min: {}, depth_max: {}'.format(depth_min, depth_max))
        depth_norm = (depth - depth_min) / (depth_max - depth_min)
        depth_vis = colorize_depth_maps(depth_norm, 0., 1.)[0].transpose((1, 2, 0))
        img_set = {'depth_vis': depth_vis, 'rgb': rgb}
        # if 'confidence' in output:
        #     confidence = output['confidence'][0, 0].detach().cpu().numpy()
        #     confidence_vis = colorize_depth_maps(confidence, 0., 1.)[0].transpose((1, 2, 0))
        #     img_set.update({'confidence_vis': confidence_vis})
        # if 'orig_size' in batch:
        #     h, w = batch['orig_size'][0][0].item(), batch['orig_size'][1][0].item()
        #     for k in img_set: img_set[k] = cv2.resize(img_set[k], (w, h), interpolation=cv2.INTER_AREA)
        rgb_name = batch['meta']['rgb_name'][0]
        img_path = join(self.output_dir, '{}'.format(rgb_name))
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        imageio.imwrite(img_path, (np.concatenate([img_set[k] for k in img_set], axis=1) * 255.).astype(np.uint8))
        img_path = img_path + '_depth.png'
        imageio.imwrite(img_path, (depth * 255.).astype(np.uint8))
        # for k, v in img_set.items():
        #     img_path = join(self.output_dir, 'images/{}_{}.jpg'.format(rgb_name.split('.')[0], k))
        #     os.makedirs(os.path.dirname(img_path), exist_ok=True)
        #     imageio.imwrite(img_path, (v * 255).astype(np.uint8))
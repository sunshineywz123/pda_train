import os
import cv2
import torch
import imageio
import numpy as np
from einops import rearrange
from typing import Any, Dict
import pytorch_lightning as pl
from os.path import join, exists
from hydra.utils import instantiate

from lib.utils.pylogger import Log
from lib.utils.depth_utils import Normalization, denormalize_depth
from lib.utils.dpt.eval_utils import recover_metric_depth, evaluate_rel_err

from easyvolcap.utils.data_utils import save_image
from easyvolcap.utils.easy_utils import write_camera


class MVSModel(pl.LightningModule):
    def __init__(self,
                 pipeline,
                 optimizer=None,
                 scheduler_cfg=None,
                 args=None,
                 ignored_weights_prefix=["pipeline.text_encoder", "pipeline.vae"],
                 **kwargs,
                 ):
        super().__init__()
        self.pipeline = instantiate(pipeline, _recursive_=False)
        self.optimizer = instantiate(optimizer)
        self.scheduler_cfg = scheduler_cfg

        # Options
        self.args = args
        self.seed = args.seed
        self.rng_type = args.rng_type  # One of [const, random, seed_plus_idx]
        self.ignored_weights_prefix = ignored_weights_prefix
        self.output_dir = join(args.get('output_dir'), args.get('save_tag'))

        # The test step is the same as validation
        self.test_step = self.validation_step

    def training_step(self, batch, batch_idx):
        # Forward and compute loss
        B = self.trainer.train_dataloader.batch_size
        outputs = self.pipeline.forward_train(batch)
        loss = outputs["loss"]
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=B)

        return outputs

    def predict_step(self, batch, batch_idx, dataloader_idx=None) -> None:
        # Task-specific predict step. CAP or GEN
        outputs = self.pipeline.forward_test(batch)
        self.save_imgs(batch, outputs, test=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        # Task-specific validation step. CAP or GEN
        outputs = self.pipeline.forward_test(batch)
        self.save_imgs(batch, outputs)

        metrics = self.compute_metrics(batch, outputs)
        B = len(batch['rgb'])
        for k, v in metrics.items():
            self.log(f"val/{k}", v, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=B)

    def compute_metrics(self, batch, outputs):
        # Get the depth normalization type
        depth_type = Normalization[batch['meta']['depth_norm'][0]]

        # Fetch and denormalize the predicted and ground truth depth map
        gt_dpt = denormalize_depth(depth_type, batch['dpt'], batch['n'], batch['f'])  # (B, 1, H, W)
        pd_dpt = denormalize_depth(depth_type, outputs['dpt'], batch['n'], batch['f'])  # (B, 1, H, W)

        # Convert to cpu and numpy
        gt_dpt = gt_dpt[:, 0].detach().cpu().numpy()  # (B, H, W)
        pd_dpt = pd_dpt[:, 0].detach().cpu().numpy()  # (B, H, W)

        # Mask
        mski = np.zeros_like(gt_dpt[0]).astype(np.uint8)  # (H, W)
        if 'msk' in batch and batch['msk'].float().mean() < 0.999:
            mski = ~batch['msk'][0, 0].detach().cpu().numpy().astype(np.bool_)  # (H, W)
        else:
            # What is this?
            mski[:45,], mski[:, :41], mski[471:], mski[:, 601:] = 1, 1, 1, 1
            mski = mski.astype(np.bool_)

        metrics = {}
        for b in range(len(gt_dpt)):
            metric = evaluate_rel_err(pd_dpt[b], gt_dpt[b], mski, scale=1.0)
            for k, v in metric.items():
                if k not in metrics:
                    metrics[k] = []
                metrics[k].append(v)

        # Average each type of the metric over the batch dimension
        for k, v in metrics.items():
            metrics[k] = np.mean(v)

        return metrics

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

    # Utils function
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        for ig_keys in self.ignored_weights_prefix:
            Log.debug(f"Remove key `{ig_keys}' from checkpoint.")
            for k in list(checkpoint["state_dict"].keys()):
                if k.startswith(ig_keys):
                    checkpoint["state_dict"].pop(k)
        super().on_save_checkpoint(checkpoint)

    # Utils function
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

    # Utils function
    def save_img(self, pred, rgb, name):
        # Save the depth map and RGB image for visualization
        from lib.utils.vis_utils import colorize_depth_maps
        depth_min, depth_max = pred.min(), pred.max()
        depth_nrm = (pred - depth_min) / (depth_max - depth_min)
        depth_vis = colorize_depth_maps(depth_nrm, 0., 1.)[0].transpose((1, 2, 0))

        # Save the normalized depth map and RGB image
        img_sets = {'depth_vis': depth_vis, 'rgb': rgb}
        img_path = join(self.output_dir, '{}'.format(name))
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        imageio.imwrite(img_path, (np.concatenate([img_sets[k] for k in img_sets], axis=1) * 255.).astype(np.uint8))

    # Utils function
    def save_raw(self, dpt, K, H, W, R, T, name):
        # Create the output directories
        raw_dpt_path = join(self.output_dir, 'raw')
        os.makedirs(raw_dpt_path, exist_ok=True)
        raw_cam_path = join(self.output_dir, 'cam', name.replace('.png', '').replace('.jpg', ''))
        os.makedirs(raw_cam_path, exist_ok=True)

        # Save the raw depth map
        save_image(join(raw_dpt_path, name.replace('.png', '.exr').replace('.jpg', '.exr')), dpt[..., None])  # (H, W, 1)
        # Save the camera parameters
        camera = dict()
        camera['000000'] = dict()
        cam = camera['000000']
        cam['K'] = K.detach().cpu().numpy()
        cam['H'] = H.detach().cpu().numpy()
        cam['W'] = W.detach().cpu().numpy()
        cam['R'] = R.detach().cpu().numpy()
        cam['T'] = T.detach().cpu().numpy()
        write_camera(camera, raw_cam_path)

    # Utils function            
    def save_imgs(self, batch, output, test=False):
        # Process the batch
        rgbs = batch['rgb'].permute(0, 2, 3, 1).detach().cpu().numpy()  # (B, H, W, 3)
        dpts = denormalize_depth(Normalization[batch['meta']['depth_norm'][0]], output['dpt'], batch['n'], batch['f'])  # (B, 1, H, W)
        dpts = dpts[:, 0].detach().cpu().numpy()  # (B, H, W)

        # Save the depth map and RGB image for visualization
        for b in range(len(rgbs)):
            self.save_img(dpts[b], rgbs[b], batch['meta']['rgb_name'][b])
            if not test:
                self.save_raw(dpts[b], batch['K'][b], batch['H'][b], batch['W'][b], batch['R'][b], batch['T'][b], batch['meta']['rgb_name'][b])

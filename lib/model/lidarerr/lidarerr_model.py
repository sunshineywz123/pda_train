from typing import Any, Dict, List
import hydra
from omegaconf import DictConfig
from sklearn import metrics
import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
from lib.utils.pylogger import Log
from os.path import join
import os
import imageio
import numpy as np
import cv2
import json
from torch.optim import AdamW, Adam

from lib.utils.dpt.eval_utils import recover_metric_depth, evaluate_rel_err, recover_metric_depth_lowres, recover_metric_depth_ransac
from einops import rearrange


class LidarErrModel(pl.LightningModule):
    def __init__(
        self,
        pipeline=None,
        optimizer=None,
        lr_table=None,
        scheduler_cfg=None,
        args=None,
        ignored_weights_prefix=["pipeline.text_encoder",
                                "pipeline.vae"],
        **kwargs,
    ):
        super().__init__()
        self.pipeline = instantiate(pipeline, _recursive_=False)
        self.optimizer = instantiate(optimizer)
        # self.optimizer = optimizer
        self.lr_table = instantiate(lr_table)
        self.scheduler_cfg = scheduler_cfg
        self.ignored_weights_prefix = ignored_weights_prefix

        self.output_dir = join(args.get('output_dir'), args.get('save_tag'))
        self.result_dict = {}
        self.test_step = self.validation_step

    def training_step(self, batch, batch_idx):
        # forward and compute loss
        outputs = self.pipeline.forward_train(batch)

        if not isinstance(self.trainer.train_dataloader, List):
            B = self.trainer.train_dataloader.batch_size
        else:
            B = self.trainer.train_dataloader[0].batch_size + \
                self.trainer.train_dataloader[1].batch_size

        loss = outputs["loss"]
        self.log("train/loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=B)
        return outputs

    def predict_step(self, batch, batch_idx, dataloader_idx=None) -> None:
        raise NotImplementedError
        # outputs = self.pipeline.forward_test(batch)
        # return outputs

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        """Task-specific validation step. CAP or GEN."""
        outputs = self.pipeline.forward_test(batch)
        metrics_dict = self.compute_metrics(batch, outputs)
        B = len(batch['rgb'])
        for b in range(B):
            self.result_dict[batch['meta']['rgb_name'][b]] = {
                k: v[b] for k, v in metrics_dict.items()}
        for k, v in metrics_dict.items():
            self.log(f"val/{k}", np.mean(v), on_step=True, on_epoch=True,
                     prog_bar=True, logger=True, batch_size=B, sync_dist=True)

    def on_validation_epoch_end(self, **kwargs):
        super().on_validation_epoch_end()
        output_path = join(
            self.output_dir, 'metrics/00_{:08d}_perimg_metrics.json'.format(self.global_step))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            for key, value in self.result_dict.items():
                item_str = json.dumps({key: value}, separators=(', ', ': '))
                f.write(item_str + '\n')
        # Log.info(f"Metrics for each image saved to {output_path}")

        summary_results = {}
        for k, v in self.result_dict.items():
            for kk, vv in v.items():
                if kk not in summary_results:
                    summary_results[kk] = []
                summary_results[kk].append(vv)
        output_path = join(
            self.output_dir, 'metrics/00_{:08d}_summary_metrics.json'.format(self.global_step))
        with open(output_path, 'w') as f:
            for key, value in summary_results.items():
                item_str = json.dumps(
                    {key: np.mean(value)}, separators=(', ', ': '))
                f.write(item_str + '\n')
        # Log.info(f"Metrics for all images saved to {output_path}")
        self.result_dict = {}

    def compute_metrics(self, batch, outputs):
        lowres_dpt = batch['lowres_dpt'][:, 0].detach().cpu().numpy()
        dpt = batch['dpt'][:, 0].detach().cpu().numpy()
        msk = dpt > 1e-3

        pred_lowres_dpt = outputs['lowres_dpt'][:, 0].detach().cpu().numpy()
        pred_dpt = outputs['dpt'][:, 0].detach().cpu().numpy()
        pred_err = outputs['err'][:, 0].detach().cpu().numpy()

        correct_dpt_predlowres_err = pred_lowres_dpt + pred_err
        correct_dpt_gtlowres_err = lowres_dpt + pred_err
        correct_dpt_predlowres_diff = pred_lowres_dpt + pred_dpt - pred_lowres_dpt
        correct_dpt_gtlowres_diff = lowres_dpt + pred_dpt - pred_lowres_dpt

        # 统计 pred_lowres_dpt的质量, 和dpt和lowres_dpt
        # 统计 gt_dpt的质量, 和dpt
        # 统计 纠正后lowres_dpt 的质量
        # 统计 pred_lowres_dpt 和 pred_dpt difference的分布

        metrics_dict = {
            'gtlowres_gtdpt': [],
            'predlowres_gtdpt': [],
            'predlowres_gtlowres': [],
            'preddpt_gtdpt': [],
            'correct_pred_err': [],
            'correct_pred_diff': [],
            'correct_gt_err': [],
            'correct_gt_diff': [],
        }

        for b in range(len(lowres_dpt)):
            metrics_dict['gtlowres_gtdpt'].append(
                float(np.abs(lowres_dpt[b][msk[b]] - dpt[b][msk[b]]).mean()))
            metrics_dict['predlowres_gtdpt'].append(
                float(np.abs(pred_lowres_dpt[b][msk[b]] - dpt[b][msk[b]]).mean()))
            metrics_dict['predlowres_gtlowres'].append(
                float(np.abs(pred_lowres_dpt[b][msk[b]] - lowres_dpt[b][msk[b]]).mean()))
            metrics_dict['preddpt_gtdpt'].append(
                float(np.abs(pred_dpt[b][msk[b]] - dpt[b][msk[b]]).mean()))
            metrics_dict['correct_pred_err'].append(
                float(np.abs(correct_dpt_predlowres_err[b][msk[b]] - dpt[b][msk[b]]).mean()))
            metrics_dict['correct_pred_diff'].append(
                float(np.abs(correct_dpt_predlowres_diff[b][msk[b]] - dpt[b][msk[b]]).mean()))
            metrics_dict['correct_gt_err'].append(
                float(np.abs(correct_dpt_gtlowres_err[b][msk[b]] - dpt[b][msk[b]]).mean()))
            metrics_dict['correct_gt_diff'].append(
                float(np.abs(correct_dpt_gtlowres_diff[b][msk[b]] - dpt[b][msk[b]]).mean()))

            from lib.utils.vis_utils import colorize_depth_maps
            _img = batch['rgb'][b].detach().cpu().numpy().transpose(1, 2, 0)
            _orig_dpt = colorize_depth_maps(dpt[b], dpt[b].min(), dpt[b].max())[
                0].transpose(1, 2, 0)
            _lowres_dpt = colorize_depth_maps(
                lowres_dpt[b], lowres_dpt[b].min(), lowres_dpt[b].max())[0].transpose(1, 2, 0)

            _pred_dpt = colorize_depth_maps(
                pred_dpt[b], pred_dpt[b].min(), pred_dpt[b].max())[0].transpose(1, 2, 0)
            _pred_lowres_dpt = colorize_depth_maps(pred_lowres_dpt[b], pred_lowres_dpt[b].min(
            ), pred_lowres_dpt[b].max())[0].transpose(1, 2, 0)
            _pred_err = colorize_depth_maps(
                pred_err[b], pred_err[b].min(), pred_err[b].max())[0].transpose(1, 2, 0)

            save_img = np.concatenate([_img, _orig_dpt, _lowres_dpt], axis=1)
            save_img = np.concatenate([save_img, np.concatenate(
                [_pred_err, _pred_dpt, _pred_lowres_dpt], axis=1)], axis=0)
            save_path = join(
                self.output_dir, 'vis/{}'.format(batch['meta']['rgb_name'][b]))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            imageio.imwrite(save_path, (save_img * 255.).astype(np.uint8))
        return metrics_dict

    def configure_optimizers(self):
        group_table = {}
        params = []
        for k, v in self.pipeline.named_parameters():
            if v.requires_grad:
                group, lr = self.lr_table.get_lr(k)
                if group not in group_table:
                    group_table[group] = len(group_table)
                    params.append({'params': [v], 'lr': lr, 'name': group})
                else:
                    params[group_table[group]]['params'].append(v)
        optimizer = self.optimizer(params=params)
        if self.scheduler_cfg is None:
            return optimizer
        scheduler_cfg = self.scheduler_cfg
        scheduler_cfg["scheduler"] = instantiate(
            scheduler_cfg["scheduler"], optimizer=optimizer)
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

    def save_img(self, dpt, rgb, name, tag):
        from lib.utils.vis_utils import colorize_depth_maps
        depth_min, depth_max = dpt.min(), dpt.max()
        depth_norm = (dpt - depth_min) / (depth_max - depth_min)
        depth_vis = colorize_depth_maps(depth_norm, 0., 1.)[
            0].transpose((1, 2, 0))
        img_path = join(self.output_dir, '{}/{}'.format(tag, name))
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        imageio.imwrite(img_path, (np.concatenate(
            [rgb, depth_vis], axis=1) * 255.).astype(np.uint8))

    def save_vis_dpt(self, dpt, name, tag):
        from lib.utils.vis_utils import colorize_depth_maps
        depth_min, depth_max = dpt.min(), dpt.max()
        depth_norm = (dpt - depth_min) / (depth_max - depth_min)
        depth_vis = colorize_depth_maps(depth_norm, 0., 1.)[
            0].transpose((1, 2, 0))
        img_path = join(self.output_dir, f'{tag}/{name}')
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        imageio.imwrite(img_path, (depth_vis * 255.).astype(np.uint8))

    def save_orig_dpt(self, dpt, name, tag):
        name = name[:-4] + '.npz'
        img_path = join(self.output_dir, f'{tag}/{name}')
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        np.savez_compressed(img_path, data=np.round(dpt, 3))

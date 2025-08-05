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
import json

from lib.utils.dpt.eval_utils import recover_metric_depth, evaluate_rel_err, recover_metric_depth_lowres, recover_metric_depth_lowres_ransac, recover_metric_depth_ransac
from einops import rearrange

class MdeModel(pl.LightningModule):
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
        if self.args.get('fit_ransac', False):
            self.recover_metric_depth = recover_metric_depth_ransac
        else:
            self.recover_metric_depth = recover_metric_depth
        # args 
        # fit_ransac: bool = False
        # args参数
        # images save
        # save_vis_pred: bool = False, 保存预测可视化结果
        # save_orig_pred: bool = False, 保存原始预测结果
        # save_align_pred: bool = False, 保存对齐gt后的结果
        # save_vis_gt: bool = False, 保存gt可视化结果
        # save_orig_gt: bool = False, 保存gt原始结果
        # save_lowres_depth: bool = False, 保存低分辨率深度图
        # save_lowres_upsampled_gt: bool = False, 保存高分辨率深度图
        # metric
        # dataset_type: [nyu, general] = general, 默认情况下为general
        # compute_rel_metric: bool = True, 是否计算相对误差
        # compute_abs_metric: bool = True, 是否计算绝对误差
        # align_pred_to_lowres: bool = True, 是否使用lowres depth对齐pred
        # compute_lowres_rel_metric: bool = False, 是否计算lowres depth的相对误差
        # compute_lowres_abs_metric: bool = False, 是否计算lowres depth的绝对误差
        
        self.ignored_weights_prefix = ignored_weights_prefix
        self.output_dir = join(args.get('output_dir'), args.get('save_tag'))
        
        self.result_dict = {}

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
        # self.save_imgs(batch, outputs)
        return outputs

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        """Task-specific validation step. CAP or GEN."""
        outputs = self.predict_step(batch, batch_idx, dataloader_idx)
        metrics_dict = self.compute_metrics(batch, outputs)
        B = len(batch['rgb'])
        # 保存每张图片的metric
        for b in range(B):
            self.result_dict[batch['meta']['rgb_name'][b]] = { k: v[b] for k, v in metrics_dict.items()}
        for k, v in metrics_dict.items():
            self.log(f"val/{k}", np.mean(v), on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=B)
            
    def on_validation_epoch_end(self, **kwargs):
        super().on_validation_epoch_end()
        output_path = join(self.output_dir, 'metrics/00_{:08d}_perimg_metrics.json'.format(self.global_step))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            for key, value in self.result_dict.items():
                item_str = json.dumps({key: value}, separators=(', ', ': '))
                f.write(item_str + '\n')
        Log.info(f"Metrics for each image saved to {output_path}")
        
        # compute summary metrics 
        summary_results = {}
        for k, v in self.result_dict.items():
            for kk, vv in v.items():
                if kk not in summary_results:
                    summary_results[kk] = []
                summary_results[kk].append(vv)
        output_path = join(self.output_dir, 'metrics/00_{:08d}_summary_metrics.json'.format(self.global_step))
        with open(output_path, 'w') as f:
            for key, value in summary_results.items():
                item_str = json.dumps({key: np.mean(value)}, separators=(', ', ': '))
                f.write(item_str + '\n')
        Log.info(f"Metrics for all images saved to {output_path}")
        self.result_dict = {}
    
    def compute_metrics(self, batch, outputs):
        depth_type = batch['meta']['depth_norm'][0]
        gt_dpt = batch['dpt'][:, 0].detach().cpu().numpy()
        metrcs_return = {}
        for b in range(len(gt_dpt)):
            
            
            if self.args.get('align_pred_to_lowres', True) and 'conf' in batch:
                pred_dpt = outputs['dpt'][b, 0].detach().cpu().numpy()
                pred_dpt = recover_metric_depth_lowres_ransac(pred_dpt, batch, b)
                if self.args.get('save_vis_pred', False):
                    self.save_vis_dpt(pred_dpt, batch['meta']['rgb_name'][b], tag='vis_pred')
                if self.args.get('save_orig_pred', False):
                    self.save_orig_dpt(pred_dpt, batch['meta']['rgb_name'][b], tag='orig_pred')
                    
            # get invalid msk
            mask_invalid = np.zeros_like(gt_dpt[0]).astype(np.uint8)
            if self.args.get('dataset_type', 'general') == 'nyu':
                # eigen crop
                mask_invalid[:45,] = 1
                mask_invalid[:, :41] = 1
                mask_invalid[471:] = 1
                mask_invalid[:, 601:] = 1
                mask_invalid = mask_invalid.astype(np.bool_)
            elif self.args.get('dataset_type', 'general') == 'general':
                mask_invalid = ~batch['msk'][b, 0].detach().cpu().numpy().astype(np.bool_)
            
            # recover absolute depth metrics
            if self.args.get('compute_abs_metric', True):
                pred_dpt = outputs['dpt'][b, 0].detach().cpu().numpy()
                if self.args.get('align_pred_to_lowres', True) and 'conf' in batch:
                    pred_dpt = recover_metric_depth_lowres(pred_dpt, batch, b, mask0=~mask_invalid)
                
                abs_metrics_dict = evaluate_rel_err(pred_dpt, gt_dpt[b], mask_invalid)
                abs_metrics_dict = {'abs_' + k: v for k, v in abs_metrics_dict.items()}
                for k, v in abs_metrics_dict.items():
                    if k not in metrcs_return:
                        metrcs_return[k] = []
                    metrcs_return[k].append(v)
                
            # recover relative depth metrics
            if self.args.get('compute_rel_metric', True):
                pred_dpt = outputs['dpt'][b, 0].detach().cpu().numpy()
                if self.args.get('align_pred_to_lowres', True) and 'conf' in batch:
                    pred_dpt = recover_metric_depth_lowres(pred_dpt, batch, b, mask0=~mask_invalid)
                aligned_depth = self.recover_metric_depth(pred_dpt, gt_dpt[b], mask0=~mask_invalid)
                rel_metrics_dict = evaluate_rel_err(aligned_depth, gt_dpt[b], mask_invalid)
                rel_metrics_dict = {'rel_' + k: v for k, v in rel_metrics_dict.items()}
                for k, v in rel_metrics_dict.items():
                    if k not in metrcs_return:
                        metrcs_return[k] = []
                    metrcs_return[k].append(v)
                if self.args.get('save_vis_pred', False):
                    self.save_vis_dpt(pred_dpt, batch['meta']['rgb_name'][b], tag='vis_pred')
                if self.args.get('save_orig_pred', False): 
                    self.save_orig_dpt(pred_dpt, batch['meta']['rgb_name'][b], tag='orig_pred')
                if self.args.get('save_align_pred', False): 
                    self.save_orig_dpt(aligned_depth, batch['meta']['rgb_name'][b], tag='align_pred')
                if self.args.get('save_vis_gt', False): 
                    self.save_vis_dpt(gt_dpt[b], batch['meta']['rgb_name'][b], tag='vis_gt')
                if self.args.get('save_orig_gt', False):
                    self.save_orig_dpt(gt_dpt[b], batch['meta']['rgb_name'][b], tag='orig_gt')
                    
                    
            if self.args.get('compute_lowres_abs_metric', False) and ('lowres_dpt' in batch or 'lowres_dpt' in outputs):
                if 'lowres_dpt' in outputs: lowres_dpt = outputs['lowres_dpt'][b, 0].detach().cpu().numpy()
                else: lowres_dpt = batch['lowres_dpt'][b, 0].detach().cpu().numpy()
                
                h, w = gt_dpt[b].shape[:2]
                highres_dpt = cv2.resize(lowres_dpt, (w, h), interpolation=cv2.INTER_LINEAR)
                
                abs_metrics_dict = evaluate_rel_err(highres_dpt, gt_dpt[b], mask_invalid)
                abs_metrics_dict = {'lowres_abs_' + k: v for k, v in abs_metrics_dict.items()}
                for k, v in abs_metrics_dict.items():
                    if k not in metrcs_return:
                        metrcs_return[k] = []
                    metrcs_return[k].append(v)
                    
            if self.args.get('compute_lowres_rel_metric', False):
                if 'lowres_dpt' in outputs: lowres_dpt = outputs['lowres_dpt'][b, 0].detach().cpu().numpy()
                else: lowres_dpt = batch['lowres_dpt'][b, 0].detach().cpu().numpy()
                
                h, w = gt_dpt[b].shape[:2]
                highres_dpt = cv2.resize(lowres_dpt, (w, h), interpolation=cv2.INTER_LINEAR)
                aligned_depth = self.recover_metric_depth(highres_dpt, gt_dpt[b], mask0=~mask_invalid) 
                rel_metrics_dict = evaluate_rel_err(aligned_depth, gt_dpt[b], mask_invalid)
                rel_metrics_dict = {'lowres_rel_' + k: v for k, v in rel_metrics_dict.items()}
                for k, v in rel_metrics_dict.items():
                    if k not in metrcs_return:
                        metrcs_return[k] = []
                    metrcs_return[k].append(v)
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
        
    def save_vis_dpt(self, dpt, name, tag):
        from lib.utils.vis_utils import colorize_depth_maps
        depth_min, depth_max = dpt.min(), dpt.max()
        depth_norm = (dpt - depth_min) / (depth_max - depth_min)
        depth_vis = colorize_depth_maps(depth_norm, 0., 1.)[0].transpose((1, 2, 0))
        img_path = join(self.output_dir, f'{tag}/{name}')
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        imageio.imwrite(img_path, (depth_vis * 255.).astype(np.uint8))
        
    def save_orig_dpt(self, dpt, name, tag):
        name = name[:-4] + '.npz'
        img_path = join(self.output_dir, f'{tag}/{name}')
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        np.savez_compressed(img_path, data=np.round(dpt, 3))
            
    def save_imgs(self, batch, output):
        return
        rgb = batch['rgb'].permute(0, 2, 3, 1).detach().cpu().numpy()
        depth = output['dpt'][:, 0].detach().cpu().numpy()
        gt = batch['dpt'][:, 0].detach().cpu().numpy()
        for b in range(len(rgb)):
            self.save_img(depth[b], rgb[b], batch['meta']['rgb_name'][b])
            if self.args.get('save_gt', False):
                self.save_img(gt[b], rgb[b], '.'.join(batch['meta']['rgb_name'][b].split('.')[:-1]) + '_gt.' + batch['meta']['rgb_name'][b].split('.')[-1])
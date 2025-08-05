from typing import Any, Dict, List
import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
from lib.utils.pylogger import Log
from os.path import join
import os
import imageio
import numpy as np
import json


class Model(pl.LightningModule):
    def __init__(
        self,
        pipeline,  # The pipeline is the model itself
        optimizer,  # The optimizer is the optimizer used to train the model
        lr_table,  # The lr_table is the learning rate table
        output_dir: str,
        output_tag: str = 'default',
        clear_output_dir=False,
        scheduler_cfg=None,  # The scheduler_cfg is the scheduler configuration
        ignored_weights_prefix=["pipeline.text_encoder",
                                "pipeline.vae"],
        **kwargs,
    ):
        super().__init__()
        self.pipeline = instantiate(pipeline, _recursive_=False)
        self.optimizer = instantiate(optimizer)
        self.lr_table = instantiate(lr_table)
        self.scheduler_cfg = scheduler_cfg

        if clear_output_dir:
            Log.warn(f"Clear output dir: {join(output_dir, output_tag)}")
            os.system(f"rm -rf {join(output_dir, output_tag)}")
        self.output_dir = join(output_dir, output_tag)
        self.metrics_dict = {}
        # The ignored_weights_prefix is the prefix of the weights that should be ignored
        self.ignored_weights_prefix = ignored_weights_prefix

        self.test_step = self.validation_step

    def training_step(self, batch, batch_idx):
        output = self.pipeline.forward_train(batch)
        if not isinstance(self.trainer.train_dataloader, List):
            B = self.trainer.train_dataloader.batch_size
        else:
            B = np.sum(
                [dataloader.batch_size for dataloader in self.trainer.train_dataloader])
        loss = output['loss']
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            import ipdb; ipdb.set_trace()
            raise ValueError(f"Loss is NaN or Inf: {loss}")
        self.log('train/loss', loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=B, sync_dist=True)
        # Log other metrics
        for k, v in output.items():
            if k != 'loss' and k.endswith('_vis'):
                self.log(f'train/{k[:-4]}', v, on_step=True,
                         on_epoch=True, prog_bar=True, logger=True, batch_size=B, sync_dist=True)
        return output

    def predict_step(self, batch, batch_idx, dataloader_idx=None) -> None:
        # 把所有的原始预测结果存下来
        raise NotImplementedError

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        """Task-specific validation step. CAP or GEN."""
        # 计算所关注的指标
        raise NotImplementedError

    # ============== Utils ================= #
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

import os

import hydra
import ptvsd
import pytorch_lightning as pl
from omegaconf import DictConfig

from lib.entrys.utils import (delete_output_dir, get_callbacks, get_data,
                              get_model, print_cfg)
from lib.utils.net_utils import find_last_ckpt_path, load_pretrained_model
from lib.utils.pylogger import Log

if 0:
    ptvsd.enable_attach(address=('0.0.0.0', 5691))

def train_net(cfg: DictConfig) -> None:
    """
    Instantiate the trainer, and then train the model.
    """
    if cfg.print_cfg: print_cfg(cfg, use_rich=True)
    callbacks = get_callbacks(cfg)
    # import ipdb;ipdb.set_trace()
    logger = hydra.utils.instantiate(cfg.logger, _recursive_=False)
    trainer = pl.Trainer(
        accelerator="gpu",
        logger=logger if logger is not None else False,
        callbacks=callbacks,
        **cfg.pl_trainer,
    )
    # seed everything before loading data
    pl.seed_everything(cfg.seed)
    # pl.seed_everything(cfg.seed + trainer.node_rank*8 + trainer.local_rank)
    datamodule: pl.LightningDataModule = get_data(cfg)
    model: pl.LightningModule = get_model(cfg)
    
    # load pretrained model
    delete_output_dir(cfg.resume_training, cfg.output_dir, cfg.confirm_delete_previous_dir)
    ckpt_path = find_last_ckpt_path(cfg.callbacks.model_checkpoint.dirpath)
    load_pretrained_model(model, ckpt_path)
    
    # training loop
    trainer.fit(model, datamodule, ckpt_path=None)
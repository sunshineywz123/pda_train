import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from lib.utils.pylogger import Log
from lib.utils.net_utils import load_pretrained_model, find_last_ckpt_path
from lib.entrys.utils import get_data, get_model, get_callbacks, print_cfg, delete_output_dir
import os


def train_net(cfg: DictConfig) -> None:
    """
    Instantiate the trainer, and then train the model.
    """
    if cfg.print_cfg: print_cfg(cfg, use_rich=True)
    callbacks = get_callbacks(cfg)
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
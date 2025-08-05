import hydra
import pytorch_lightning as pl

from omegaconf import DictConfig
from lib.utils.net_utils import load_pretrained_model, find_last_ckpt_path
from lib.entrys.utils import get_data, get_model, get_callbacks, print_cfg
from typing import Tuple
from tqdm.auto import tqdm
from lib.utils.pylogger import Log

def setup_trainer(cfg: DictConfig) -> Tuple[pl.Trainer, pl.LightningModule, pl.LightningDataModule]:
    """
    Set up the PyTorch Lightning trainer, model, and data module.
    """
    if cfg.print_cfg: print_cfg(cfg, use_rich=True)
    pl.seed_everything(cfg.seed)
    # preparation
    datamodule = get_data(cfg, wo_train=True)
    model = get_model(cfg)
    ckpt_path = find_last_ckpt_path(cfg.callbacks.model_checkpoint.dirpath)
    load_pretrained_model(model, ckpt_path)

    # PL callbacks and logger
    callbacks = get_callbacks(cfg)
    logger = hydra.utils.instantiate(cfg.logger, _recursive_=False)

    # PL-Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        logger=logger if logger is not None else False,
        callbacks=callbacks,
        **cfg.pl_trainer,
    )

    return trainer, model, datamodule

def val(cfg: DictConfig) -> None:
    """
    Validate the model.
    """
    trainer, model, datamodule = setup_trainer(cfg)
    trainer.validate(model, datamodule.val_dataloader())

def predict(cfg: DictConfig) -> None:
    """
    Predict using the model.
    """
    trainer, model, datamodule = setup_trainer(cfg)
    trainer.predict(model, datamodule.val_dataloader())

def test(cfg: DictConfig) -> None:
    """
    Test the model.
    """
    trainer, model, datamodule = setup_trainer(cfg)
    trainer.test(model, datamodule.test_dataloader())
    
def debug_train_dataloader(cfg: DictConfig) -> None:
    """
    Debug the training dataloader.
    """
    # trainer, model, datamodule = setup_trainer(cfg)
    datamodule = get_data(cfg)
    dataloader = datamodule.train_dataloader()
    for data in tqdm(iter(dataloader)):
        # print(data[0][0]['image_name'], data[0][0]['mask'].sum())
        # print(data[0][1]['image_name'], data[0][1]['mask'].sum())
        pass
        # Log.info(data.keys())

def debug_val_dataloader(cfg: DictConfig) -> None:
    """
    Debug the training dataloader.
    """
    # trainer, model, datamodule = setup_trainer(cfg)
    datamodule = get_data(cfg, wo_train=True)
    dataloader = iter(datamodule.val_dataloader())
    for data in tqdm(dataloader):
        pass
        # Log.info(data.keys())
        
def debug_cfg(cfg: DictConfig) -> None:
    """
    Debug the config.
    """
    print_cfg(cfg, use_rich=True)
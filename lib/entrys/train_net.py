import os

import hydra
import ptvsd
import pytorch_lightning as pl
from omegaconf import DictConfig

from lib.entrys.utils import (delete_output_dir, get_callbacks, get_data,
                              get_model, print_cfg)
from lib.utils.net_utils import find_last_ckpt_path, load_pretrained_model
from lib.utils.pylogger import Log
import os
try: import requests
except: os.system('pip install requests'); import requests
try: from dingtalkchatbot.chatbot import DingtalkChatbot, ActionCard, CardItem
except: os.system('pip install DingtalkChatbot'); from dingtalkchatbot.chatbot import DingtalkChatbot, ActionCard, CardItem
# webhook = 'https://oapi.dingtalk.com/robot/send?access_token=56e7dc92fddd78bbc4d457a08d037e6d866664fd90ddf7100ee3a03882f07153'
# secret = 'SEC1f896436f9c0fffa73a95d973d65f68d0dc9092a6c6e4d016b44a8175bf61c18'
# https://oapi.dingtalk.com/robot/send?access_token=caac9d28c298db0d53a89512b6fcc173fda4e371087540cc937c2f17a711b242
# curl 'https://oapi.dingtalk.com/robot/send?access_token=caac9d28c298db0d53a89512b6fcc173fda4e371087540cc937c2f17a711b242'  -H 'Content-Type: application/json' -d '{"msgtype": "text","text": {"content":"????????????, ?????????????????????"}}'
webhook = 'https://oapi.dingtalk.com/robot/send?access_token=caac9d28c298db0d53a89512b6fcc173fda4e371087540cc937c2f17a711b242'
secret = 'SECec6827303ee1635844d871d118c8a99f96b2955b6f6181d20ea8e82db39b2472'
robot = DingtalkChatbot(webhook, secret=secret)
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
    robot.send_text(msg='hi train done!!!')
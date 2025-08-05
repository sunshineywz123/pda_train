from unittest import loader
import pytorch_lightning as pl
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from hydra.utils import instantiate, get_method
from torch.utils.data import DataLoader, ConcatDataset, Subset, Sampler, BatchSampler, DistributedSampler
from omegaconf import ListConfig, DictConfig
from lib.utils.pylogger import Log
from numpy.random import choice
import torch
import numpy as np
import torch.distributed as dist
from copy import deepcopy

class GeneralDataModule(pl.LightningDataModule):
    default_train_loader_opts = DictConfig({
        'batch_size': 1,
        'num_workers': 4,
        'shuffle': True,
        'pin_memory': True,
        'drop_last': True,
        'persistent_workers': True
    })
    default_val_loader_opts = DictConfig({
        'batch_size': 1,
        'num_workers': 1,
        'shuffle': False,
        'pin_memory': False,
        'drop_last': False,
        'persistent_workers': True
    })
    def __init__(
        self,
        train_dataset: DictConfig = None,
        val_dataset: DictConfig = None,
        test_dataset: DictConfig = None,
        train_loader_opts: DictConfig = None,
        val_loader_opts: DictConfig = None,
        **kwargs
    ):
        """This is a general datamodule that can be used for any dataset.
        Train uses ConcatDataset
        Val and Test use CombinedLoader, sequential, completely consumes ecah iterable sequentially, and returns a triplet (data, idx, iterable_idx)
        Args:
            name: used by other module
            dataset_opts: the target of the dataset. e.g. dataset_opts.train = {_target_: ..., limit_size: None}
            loader_opts: the options for the dataset
            limit_each_trainset: limit the size of each dataset, None means no limit, useful for debugging
        """
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.train_loader_opts = self.default_train_loader_opts
        self.val_loader_opts = self.default_val_loader_opts
        
        if train_loader_opts is not None: self.train_loader_opts.update(train_loader_opts)
        if val_loader_opts is not None: self.val_loader_opts.update(val_loader_opts)
       
        self.train_loader_opts.persistent_workers = True if self.train_loader_opts.num_workers > 0 else False
        self.val_loader_opts.persistent_workers = True if self.val_loader_opts.num_workers > 0 else False
        

    def val_dataloader(self):
        loaders = GeneralDataModule._parse_loaders(self.val_dataset, self.val_loader_opts)
        if isinstance(loaders, list):
            return CombinedLoader(loaders, mode='sequential')
        else:
            return loaders

    def test_dataloader(self):
        loaders = GeneralDataModule._parse_loaders(self.test_dataset, self.val_loader_opts)
        if isinstance(loaders, list):
            return CombinedLoader(loaders, mode='sequential')
        else:
            return loaders

    def train_dataloader(self):
        return GeneralDataModule._parse_train_dataloader(self.train_dataset, self.train_loader_opts)

    @staticmethod
    def _parse_train_dataloader(config, loader_opts):
        if isinstance(config.dataset_opts, ListConfig) and 'combined_loader_opts' in config:
            dataloaders = GeneralDataModule._parse_loaders(config, loader_opts)
            return CombinedLoader(dataloaders, **config.combined_loader_opts)
        elif isinstance(config.dataset_opts, ListConfig):
            datasets = GeneralDataModule._parse_datasets(config)
            dataset = ConcatDataset(datasets)
            if 'loader_opts' in config:
                loader_opts = deepcopy(loader_opts)
                loader_opts.update(config.loader_opts)
            return DataLoader(dataset, **loader_opts)
        else:
            return GeneralDataModule._parse_loaders(config, loader_opts)

    @staticmethod
    def _parse_datasets(config):
        datasets = []
        for idx, dataset_opt in enumerate(config.dataset_opts):
            dataset = instantiate(dataset_opt)
            datasets.append(dataset)
        return datasets

    @staticmethod
    def _parse_loaders(config, loader_opts):
        if not isinstance(config.dataset_opts, ListConfig):
            dataset = instantiate(config.dataset_opts)
            if 'loader_opts' in config:
                loader_opts = deepcopy(loader_opts)
                loader_opts.update(config.loader_opts)
            return DataLoader(dataset, **loader_opts)
        else:
            dataloaders = []
            for idx, dataset_opt in enumerate(config.dataset_opts):
                if isinstance(dataset_opt, ListConfig):
                    datasets = [instantiate(opt) for opt in dataset_opt]
                    dataset = ConcatDataset(datasets)
                else:
                    dataset = instantiate(dataset_opt)
                if 'loader_opts' in config:
                    loader_opt = deepcopy(loader_opts)
                    if isinstance(config.loader_opts, ListConfig):
                        loader_opt.update(config.loader_opts[idx])
                    else:
                        loader_opt.update(config.loader_opts)
                dataloaders.append(DataLoader(dataset, **loader_opt))
            return dataloaders
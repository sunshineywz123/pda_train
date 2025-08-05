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


class GeneralDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        dataset_opts: DictConfig, 
        loader_opts: DictConfig, 
        collate_fn = None, 
        wo_train: bool = False
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
        self.dataset_opts = dataset_opts
        self.loader_opts = loader_opts
        self.collate_fn = get_method(collate_fn) if collate_fn else None

    def train_dataloader(self):
        if not isinstance(self.dataset_opts.train, ListConfig):
            dataset = instantiate(self.dataset_opts.train)
            return DataLoader(
                dataset,
                collate_fn=self.collate_fn,
                **self.loader_opts.train
            )
        else:
            dataloaders = []
            for dataset_opt, loader_opt in zip(self.dataset_opts.train, self.loader_opts.train):
                dataset = instantiate(dataset_opt)
                dataloaders.append(
                    DataLoader(
                        dataset,
                        collate_fn=self.collate_fn,
                        **loader_opt
                    )
                )
            return CombinedLoader(dataloaders, mode=self.loader_opts.get('train_mode', 'max_size_cycle'))

    def val_dataloader(self):
        if not isinstance(self.dataset_opts.val, ListConfig):
            dataset = instantiate(self.dataset_opts.val)
            return DataLoader(
                dataset,
                collate_fn=self.collate_fn,
                **self.loader_opts.val
            )
        else:
            dataloaders = []
            for dataset_opt, loader_opt in zip(self.dataset_opts.val, self.loader_opts.val):
                dataset = instantiate(dataset_opt)
                dataloaders.append(
                    DataLoader(
                        dataset,
                        collate_fn=self.collate_fn,
                        **loader_opt
                    )
                )
            return CombinedLoader(dataloaders, mode="sequential")

    def test_dataloader(self):
        if not isinstance(self.dataset_opts.test, ListConfig):
            dataset = instantiate(self.dataset_opts.test)
            return DataLoader(
                dataset,
                collate_fn=self.collate_fn,
                **self.loader_opts.test
            )
        else:
            dataloaders = []
            for dataset_opt, loader_opt in zip(self.dataset_opts.test, self.loader_opts.test):
                dataset = instantiate(dataset_opt)
                dataloaders.append(
                    DataLoader(
                        dataset,
                        collate_fn=self.collate_fn,
                        **loader_opt
                    )
                )
            return CombinedLoader(dataloaders, mode="sequential")
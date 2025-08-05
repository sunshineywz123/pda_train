from curses import meta
import os
import pdbr
import json
import torch
import numpy as np
from glob import glob
import pytorch_lightning as pl
from numpy.random import choice
import torch.distributed as dist
from os.path import join, exists, dirname
from omegaconf import ListConfig, DictConfig
from hydra.utils import instantiate, get_method
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, ConcatDataset, Subset, Sampler, BatchSampler, DistributedSampler

from lib.utils.pylogger import Log


class GeneralizableDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_opts: DictConfig,
                 loader_opts: DictConfig,
                 name='test',
                 collate_fn=None,
                 limit_each_trainset=None,
                 concat_independent: bool = False,
                 wo_train: bool = False
                 ):
        """ This is a general datamodule that can be used for any dataset.
            Train uses ConcatDataset Val and Test use CombinedLoader, sequential, completely
            consumes ecah iterable sequentially, and returns a triplet (data, idx, iterable_idx)

        Args:
            name: used by other module
            dataset_opts: the target of the dataset. e.g. dataset_opts.train = {_target_: ..., limit_size: None}
            loader_opts: the options for the dataset
            limit_each_trainset: limit the size of each dataset, None means no limit, useful for debugging
        """

        super().__init__()
        self.name = name
        self.loader_opts = loader_opts
        self.collate_fn = get_method(collate_fn) if collate_fn else None
        self.limit_each_trainset = limit_each_trainset
        self.concat_independent = concat_independent

        for split in ("train", "val", "test"):
            if split not in dataset_opts: continue
            if wo_train and split == "train": continue

            # Get the split options
            split_opts = dataset_opts.get(split)

            # If there are multiple configurations for the split
            if not isinstance(split_opts, ListConfig): split_opts = [split_opts]

            dataset = []
            for split_opts_i in split_opts:
                # Get the meta dataset roots and find all the datasets under the root
                meta_roots = split_opts_i.get('meta_roots', None)
                # Load the specific meta infos if any
                meta_infos = json.load(open(split_opts_i.meta_infos)) if split_opts_i.get('meta_infos', None) else None

                # If there are no meta roots, then instantiate the dataset directly using the `data_root`
                if meta_roots is None:
                    # Modify the specific opts if there has any meta informations
                    if meta_infos is not None:
                        for k, v in meta_infos[split_opts_i.data_root].items():
                            split_opts_i[k] = v
                    dataset_i = instantiate(split_opts_i)
                    # Limit the size of the dataset
                    if self.limit_each_trainset: dataset_i = Subset(dataset_i, choice(len(dataset_i), self.limit_each_trainset))
                    dataset.append(dataset_i)

                # If there is `meta_roots`, then instantiate the dataset for each data root under the meta root
                else:
                    # If there are multiple `meta_roots`
                    if not isinstance(meta_roots, ListConfig): meta_roots = [meta_roots]

                    data_roots = []
                    # Create the dataset for each meta root
                    for meta_root in meta_roots:
                        if exists(join(meta_root, 'scenes.json')):
                            scenes = sorted(json.load(open(join(meta_root, 'scenes.json')))['scenes'])
                        else:
                            scenes = sorted(dirname(p) for p in glob(join(meta_root, "**", "extri.yml"), recursive=True))
                        data_roots.extend(scenes)

                    # Create the dataset for each data root under the meta root
                    for data_root in data_roots:
                        # Copy the split_opts_i and update the data_root
                        config = split_opts_i.copy()
                        config.data_root = data_root
                        # Modify the specific opts if there has any meta informations
                        if meta_infos is not None:
                            for k, v in meta_infos[data_root].items():
                                config[k] = v
                        # Instantiate the dataset
                        dataset_i = instantiate(config)
                        # Limit the size of the dataset
                        if self.limit_each_trainset: dataset_i = Subset(dataset_i, choice(len(dataset_i), self.limit_each_trainset))
                        # Append the dataset
                        dataset.append(dataset_i)

            # Only concatenate during training, otherwise, there will be CombinedLoader ERROR
            if split == 'train':
                # Concatenate the datasets
                dataset = ConcatDataset(dataset)
                if self.concat_independent:
                    self.train_sampler = CustomSampler(dataset.cummulative_sizes, 
                                                        loader_opts.train.batch_size, 
                                                        max_iter=loader_opts.train.get('max_iter', -1),
                                                        prob=loader_opts.train.get('dataset_split_prob', None))

                # Set and log the concatenated training dataset
                setattr(self, f"{split}set", dataset)
                Log.info(f"[Dataset]: Split={split}, Dataset size={len(dataset)}")

            else:
                # Set and log the validation and test datasets
                setattr(self, f"{split}sets", dataset)
                for dataset_i in dataset:
                    Log.info(f"[Dataset]: Split={split}, Dataset size={len(dataset_i)}, {split_opts_i._target_}")

    def train_dataloader(self):
        if hasattr(self, "trainset"):
            if self.concat_independent:
                data_loader_args = {
                    'num_workers': self.loader_opts.train.num_workers, 
                    'persistent_workers': True and self.loader_opts.train.num_workers > 0, 
                    'batch_sampler': self.train_sampler,
                    'collate_fn': self.collate_fn
                }
            else:
                data_loader_args = {
                    'shuffle': True, 
                    'num_workers': self.loader_opts.train.num_workers, 
                    'persistent_workers': True and self.loader_opts.train.num_workers > 0, 
                    'batch_size': self.loader_opts.train.batch_size, 
                    'drop_last': True, 
                    'collate_fn': self.collate_fn
                }
                
            return DataLoader(
                self.trainset,
                **data_loader_args
            )
        else:
            return super().train_dataloader()

    def val_dataloader(self):
        if hasattr(self, "valsets"):
            loaders = []
            for valset in self.valsets:
                loaders.append(
                    DataLoader(
                        valset,
                        shuffle=False,
                        num_workers=self.loader_opts.val.num_workers,
                        persistent_workers=True and self.loader_opts.val.num_workers > 0,
                        batch_size=self.loader_opts.val.batch_size,
                        collate_fn=self.collate_fn,
                    )
                )
            return CombinedLoader(loaders, mode="sequential")
        else:
            return super().val_dataloader()

    def test_dataloader(self):
        if hasattr(self, "testsets"):
            loaders = []
            for testset in self.testsets:
                loaders.append(
                    DataLoader(
                        testset,
                        shuffle=False,
                        num_workers=self.loader_opts.test.num_workers,
                        persistent_workers=False,
                        batch_size=self.loader_opts.test.batch_size,
                        collate_fn=self.collate_fn,
                    )
                )
            return CombinedLoader(loaders, mode="sequential")
        else:
            return super().test_dataloader()

class CustomSampler(BatchSampler):
    def __init__(self, cummulative_sizes, batch_size, max_iter = -1, prob = None):
        self.cummulative_sizes = cummulative_sizes
        self.data_sizes = [cummulative_sizes[i] - (cummulative_sizes[i-1] if i > 0 else 0) for i in range(len(cummulative_sizes))]
        self.dataset_prob = np.asarray(self.data_sizes) / sum(self.data_sizes) if prob is None else np.asarray(prob)
        Log.info(f'[CustomSampler]: {str(self.dataset_prob)}')
        self.batch_size = batch_size
        self.max_iter = max_iter
        # shuffle, batch_size, no drop last
        
    def __iter__(self):
        # 根据data_sizes，按照概率计算dataset_idx
        while True:
            dataset_idx = choice(range(len(self.cummulative_sizes)), p=self.dataset_prob)
            start = 0 if dataset_idx == 0 else self.cummulative_sizes[dataset_idx - 1]
            end = self.cummulative_sizes[dataset_idx]
            indices = torch.randint(start, end, (self.batch_size,))
            yield iter(indices.tolist())
        # dataset_idx = choice(range(len(self.cummulative_sizes)), p=self.dataset_prob)
        # start = 0 if dataset_idx == 0 else self.cummulative_sizes[dataset_idx - 1]
        # end = self.cummulative_sizes[dataset_idx]
        # indices = torch.randint(start, end, (self.batch_size,))
        # yield iter(indices.tolist())
        # for _ in range(self.batch_size):
        #     index = torch.randint(start, end, (1,)).item()
        #     yield index

    def __len__(self):
        return self.max_iter if self.max_iter != -1 else self.cummulative_sizes[-1] // self.batch_size

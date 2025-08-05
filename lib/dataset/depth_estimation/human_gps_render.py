from lib.dataset.depth_estimation.depth_estimation import Dataset as BaseDataset
from lib.dataset.depth_estimation.depth_estimation import *
from os.path import join
import os
from torchvision.transforms import Compose
import pandas as pd
from tqdm import tqdm
import glob
import json

class Dataset(BaseDataset):
    def build_metas(self):
        self.dataset_name = 'human_gps_render'
        self.rgb_files, self.depth_files = [], []
        split_file = json.load(open(self.cfg.split_file))
        self.rgb_files = split_file['rgb_files']
        self.depth_files = split_file['depth_files']
        frames = self.cfg.get('frames', [0, -1, 1])
        s, e, i = frames[0], frames[1], frames[2]
        e = min(e, len(self.rgb_files)) if e != -1 else len(self.rgb_files)
        self.rgb_files = self.rgb_files[s:e:i]
        self.depth_files = self.depth_files[s:e:i]
        self.all_valid = self.cfg.get('all_valid', False)
        self.normalize_depth = self.cfg.get('normalize_depth', True)
        self.split = self.cfg.get('split', 'train')

    def read_depth(self, index, depth=None):
        depth, valid_mask = super().read_depth(index, depth)
        depth = depth / (2**15/1000)
        depth[valid_mask==1] = 1 / depth[valid_mask==1]
        depth_max = depth[valid_mask==1].max() if self.split != 'train' else depth[valid_mask==1].max() * (np.random.random() + 1.)
        depth[valid_mask==0] = depth_max
        if self.normalize_depth:
            depth_min = depth[valid_mask==1].min()
            depth_max = depth[valid_mask==1].max()
            if depth_min == depth_max: depth_max = depth_min + 1e-3
            depth = (depth - depth_min) / (depth_max - depth_min)
            depth = np.clip(depth, 0, 1)
        if self.all_valid: 
            valid_mask = np.ones_like(depth).astype(np.uint8)
        return depth, valid_mask
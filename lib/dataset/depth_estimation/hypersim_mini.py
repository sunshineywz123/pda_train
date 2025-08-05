from lib.dataset.depth_estimation.depth_estimation import Dataset as BaseDataset
from lib.dataset.depth_estimation.depth_estimation import *
from os.path import join
import os
from torchvision.transforms import Compose
import json
from copy import deepcopy
from lib.utils.pylogger import Log
from tqdm import tqdm

class Dataset(BaseDataset):
        
    def read_rgb_name(self, index):
        return '__'.join(self.rgb_files[index].split('/')[-4:])
    
    def build_metas(self):
        self.dataset_name = 'hypersim_mini'
        splits = json.load(open(self.cfg.split_path))
        key = 'train_rgb_paths' if self.cfg.split == 'train' else 'test_rgb_paths'
        rgb_paths = splits[key]
        key = 'train_dpt_paths' if self.cfg.split == 'train' else 'test_dpt_paths'
        dpt_paths = splits[key]
        assert len(rgb_paths) == len(dpt_paths)
        self.rgb_files = rgb_paths
        self.depth_files = dpt_paths
        
    def build_transforms(self):
        transforms = []
        
        width = self.cfg.get('width', None)
        height = self.cfg.get('height', None)
        resize_target = self.cfg.get('resize_target', False)
        ensure_multiple_of = self.cfg.get('ensure_multiple_of', 8)
        Log.info(f"Using {self.cfg.split} width: {width}")
        Log.info(f"Using {self.cfg.split} height: {height}")
        Log.info(f"Resize {self.cfg.split} target: {resize_target}")
        Log.info(f"Using {self.cfg.split} ensure_multiple_of: {ensure_multiple_of}")
        resize_layer = Resize(
            width=width,
            height=height,
            resize_target=resize_target,
            keep_aspect_ratio=True,
            ensure_multiple_of=ensure_multiple_of,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_AREA
        )
        transforms.append(resize_layer)
        transforms.append(PrepareForNet())
        self.transform = Compose(transforms)
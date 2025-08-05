import os
from abc import ABC, abstractmethod
import hydra
from omegaconf import DictConfig
import cv2
import numpy as np
import imageio
import random
from PIL import ImageFilter
import torch
import torch.nn.functional as F
import math
from lib.utils.pylogger import Log
import time
import h5py
from torchvision.transforms import Compose

class Dataset(ABC):
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self.cfg = DictConfig(kwargs)
        self.dataset_name = self.cfg.get('dataset_name', 'unknown')
        self.use_low = self.cfg.get('use_low', True)
        self.build_metas()
        self.build_transforms()
        self.fixed_num_files = self.cfg.get('fixed_num_files', -1)
        Log.info(f'{self.cfg.split} split of {self.dataset_name} dataset: {len(self.rgb_files)} frames in total.')
        if self.fixed_num_files > 0: Log.info(f'Fixed number of files: {self.fixed_num_files}')
    #     self.max_length = self.cfg.get('max_length', -1)
    #     if self.max_length != -1 and self.max_length < len(self.rgb_files) and self.cfg.split == 'train':
    #         self.shuffle_dataset(self.max_length)
                
    # def shuffle_dataset(self, max_length):
    #     Log.info(f'Shuffle dataset with max_length: {max_length}')
    #     ids = np.arange(len(self.rgb_files))
    #     np.random.shuffle(ids)
    #     self.rgb_files = [self.rgb_files[i] for i in ids[:max_length]]
    #     self.depth_files = [self.depth_files[i] for i in ids[:max_length]]

    @abstractmethod 
    def build_metas(self):
        pass
        # depth_files
        # rgb_files
    
    def build_transforms(self):
        transforms = self.cfg.get('transforms', [])
        if len(transforms) == 0: 
            self.transform = lambda x: x
            return
        log_str = f'{self.dataset_name} transform layers: \n'
        for idx, transform in enumerate(transforms):
            log_str += (str(transform) + '\n') if idx != len(transforms) - 1 else str(transform)
        Log.info(log_str)
        self.transform = Compose(transforms)
    
    def read_rgb(self, index):
        img_path = self.rgb_files[index]
        start_time = time.time()
        rgb = cv2.imread(img_path)
        end_time = time.time()
        if end_time - start_time > 1: Log.warn(f'Long time to read {img_path}: {end_time - start_time}')
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return np.asarray(rgb / 255.).astype(np.float32)
    
    def read_rgb_name(self, index):
        return '__'.join(self.rgb_files[index].split('/')[-3:])
    
    def read_depth(self, index, depth=None):
        if not hasattr(self, 'depth_files'): return None, None
        Log.debug(index, self.depth_files[index])
        start_time = time.time()
        if depth is not None: pass
        elif self.depth_files[index].endswith('.png'):
            depth_path = self.depth_files[index]
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 1000.
        elif self.depth_files[index].endswith('.npz'):
            depth = np.load(self.depth_files[index])['data']
        elif self.depth_files[index].endswith('.hdf5'):
            depth = h5py.File(self.depth_files[index])['dataset']
            depth = np.asarray(depth)
        end_time = time.time()
        if end_time - start_time > 1: Log.warn(f'Long time to read {self.depth_files[index]}: {end_time - start_time}')
        valid_mask = np.logical_and(depth > 0.01, ~np.isnan(depth)) & (~np.isinf(depth))
        if valid_mask.sum() == 0: Log.warn('No valid mask in the depth map of {}'.format(self.depth_files[index]))
        if valid_mask.sum() != 0 and np.isnan(depth).sum() != 0: depth[np.isnan(depth)] = depth[valid_mask].max()
        if valid_mask.sum() != 0 and np.isinf(depth).sum() != 0: depth[np.isinf(depth)] = depth[valid_mask].max()
        return depth, valid_mask.astype(np.uint8)
    
    def check_shape(self, rgb, dpt):
        assert(rgb.shape[:2] == dpt.shape[:2]), "rgb.shape: {}, dpt.shape: {}".format(rgb.shape, dpt.shape)
        assert(len(rgb.shape) == 3), "rgb.shape: {}".format(rgb.shape)
        assert(len(dpt.shape) == 2), "dpt.shape: {}".format(dpt.shape)
        
        
    def read_low_depth(self, file, **kwargs):
        return np.asarray(imageio.imread(file) / 1000.).astype(np.float32)
    
    def __getitem__(self, index):
        index = index % len(self.rgb_files)
        
        repeat_num = 0
        while True:
            rgb, (dpt, msk) = self.read_rgb(index), self.read_depth(index)
            if dpt is not None:
                self.check_shape(rgb, dpt)
                # assert(rgb.shape[:2] == dpt.shape[:2]), "rgb.shape: {}, dpt.shape: {}".format(rgb.shape, dpt.shape)
                # assert(len(rgb.shape) == 3), "rgb.shape: {}".format(rgb.shape)
                # assert(len(dpt.shape) == 2), "dpt.shape: {}".format(dpt.shape)
            sample = {
                'image': rgb,
            }
            if dpt is not None:
                sample['depth'] = dpt
                sample['mask'] = msk
            
            # for depth upsampling task
            if hasattr(self, 'low_files') and self.use_low:
                
                lowres_depth = self.read_low_depth(self.low_files[index], index=index)
                # lowres_depth = np.asarray(imageio.imread(self.low_files[index]) / 1000.).astype(np.float32)
                # Log.debug('lowres_depth', lowres_depth.shape, self.low_files[index])
                if lowres_depth is not None: sample['lowres_depth'] = lowres_depth
                
            if hasattr(self, 'conf_files'):
                confidence = np.asarray(imageio.imread(self.conf_files[index]))
                sample['confidence'] = confidence

            # semantic and mesh depth
            if hasattr(self, 'sem_files'):
                sem = np.asarray(imageio.imread(self.sem_files[index]))
                sample['semantic'] = sem

            if hasattr(self, 'mesh_depth_files'):
                mesh_depth = np.asarray(imageio.imread(self.mesh_depth_files[index]) / 1000.).astype(np.float32)
                sample['mesh_depth'] = mesh_depth

            if hasattr(self, 'scene_direction'):
                scene = self.rgb_files[index].split('/')[-3]
                direction = self.scene_direction[scene]
                sample['direction'] = direction
                
            sample = self.transform(sample)
            
            if 'mask' not in sample or sample['mask'].sum() >= 10:
                break
            else:
                repeat_num += 1
                index = int(np.random.randint(0, len(self.rgb_files)))
                image_name = self.rgb_files[index]
                if repeat_num >= 1: Log.warn(f'No valid mask in the depth map of {image_name}.')
                elif repeat_num > 5: Log.warn(f'No valid mask in the depth map of {image_name}.')
                elif repeat_num > 10: raise ValueError(f'No valid mask in the depth map of {image_name}.')
                
        sample['dataset_name'] = self.dataset_name
        sample['image_name'] = self.read_rgb_name(index)
        sample['image_path'] = self.rgb_files[index]
        return sample
    
    def __len__(self):
        return self.fixed_num_files if self.fixed_num_files > 0 else len(self.rgb_files)
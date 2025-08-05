import os
import numpy as np
from tqdm.auto import tqdm
from copy import deepcopy
import imageio
import glob
import cv2
import json
from lib.utils.pylogger import Log
from omegaconf import DictConfig

class Dataset:
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self.cfg = DictConfig(kwargs)
        left_paths = sorted(glob.glob(os.path.join(self.cfg.data_root, 'outleft', '*.png')))
        img_names = [os.path.basename(p) for p in left_paths]
        right_paths = [os.path.join(self.cfg.data_root, 'outright', n) for n in img_names]
        warped_paths = [os.path.join(self.cfg.data_root, 'warped_img/depth_anything', n) for n in img_names]
        self.metas = [ (l, r, w) for l, r, w in zip(left_paths, right_paths, warped_paths)]
        
        start_idx, end_idx, itnerval = kwargs['frames']
        end_idx = len(self.metas) if end_idx == -1 else end_idx
        self.metas = self.metas[start_idx:end_idx:itnerval]
        
    def __len__(self):
        return len(self.metas)
        
    def __getitem__(self, index):
        l, r, w = self.metas[index]
        left_img = (np.asarray(imageio.imread(l)) / 255.).astype(np.float32).transpose(2, 0, 1)
        right_img = (np.asarray(imageio.imread(r)) / 255.).astype(np.float32).transpose(2, 0, 1)
        warped_img = (np.asarray(imageio.imread(w)) / 255.).astype(np.float32).transpose(2, 0, 1)
        return {
            'left_img': left_img,
            'warped_img': warped_img,
            'right_img': right_img
        }
            
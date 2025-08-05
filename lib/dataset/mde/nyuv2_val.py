import os
import numpy as np
from tqdm.auto import tqdm
from lib.dataset.mde.mde import Dataset as BaseDataset
from copy import deepcopy
import imageio
import glob
import cv2
import json
from lib.utils.pylogger import Log
import scipy.io as sio

class Dataset(BaseDataset):
    def build_metas(self):
        self.data = sio.loadmat('{}/{}/{}.mat'.format(self.cfg.get('data_root'), self.cfg.get('split'), self.cfg.get('split')))
        frame_len = len(self.data['rgbs'])
        frame_sample = deepcopy(self.cfg.frames)
        frame_sample[1] =  frame_len if frame_sample[1] == -1 else frame_sample[1]
        self.rgbs = self.data['rgbs'][frame_sample[0]:frame_sample[1]:frame_sample[2]]
        self.dpts = self.data['depths'][frame_sample[0]:frame_sample[1]:frame_sample[2]]
        self.metas = self.rgbs
        Log.info(f'[Dataset]: {len(self.rgbs)} frames in total.')
    
    def read_rgb(self, index):
        rgb = self.rgbs[index]
        rgb = (np.asarray(rgb) / 255.).astype(np.float32).transpose(2, 1, 0)
        return rgb
    
    def read_dpt(self, index):
        dpt = np.asarray(self.dpts[index]).astype(np.float32)
        # ################
        # log depth
        # dpt = np.clip(dpt, 0.5, 80.)
        # dpt = (np.log(dpt) - np.log(0.5))/(np.log(80.) - np.log(0.5))
        # ################
        
        # ################
        # official depth
        if self.cfg.get('depth_norm', 'nonorm') == 'simp':
            dpt_min, dpt_max = np.percentile(dpt, 2.), np.percentile(dpt, 98.)
            if dpt_max - dpt_min < 1e-6: dpt_max = dpt_min + 2e-6
            dpt = np.clip(dpt, dpt_min, dpt_max)
            dpt = (dpt - dpt_min) / (dpt_max - dpt_min)
        elif self.cfg.get('depth_norm', 'nonorm') == 'log':
            dpt = np.clip(dpt, 0.5, 80.)
            dpt = (np.log(dpt) - np.log(0.5))/(np.log(80.) - np.log(0.5))
        elif self.cfg.get('depth_norm', 'nonorm') == 'disp':
            dpt = np.clip(dpt, 1., None)
            dpt = 1 / dpt
        elif self.cfg.get('depth_norm', 'nonorm') == 'nonorm':
            pass
        else:
            import ipdb; ipdb.set_trace()
        # dpt = np.clip(dpt, 0.1, None)
        # dpt_min, dpt_max = np.percentile(dpt, 2.), np.percentile(dpt, 98.)
        # if dpt_max - dpt_min <= 1e-6: dpt_max = dpt_min + 2e-6
        # dpt = np.clip(dpt, dpt_min, dpt_max)
        # dpt = (dpt - dpt_min) / (dpt_max - dpt_min)
        # ################
        return dpt.astype(np.float32).transpose(1, 0), None
    
    def read_rgb_name(self, index) -> str:
        # scene + camera + frame
        return 'frame_{:04d}.jpg'.format(index)
    
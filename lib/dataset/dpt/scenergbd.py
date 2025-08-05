import os
from os.path import join
import numpy as np
from tqdm.auto import tqdm
from lib.dataset.dpt.marigold_train import Dataset as BaseDataset
from copy import deepcopy
import imageio
import glob
import cv2
import json
from lib.utils.pylogger import Log

class Dataset(BaseDataset):
    def build_metas(self):
        splits = sorted(os.listdir(join(self.cfg.data_root, self.cfg.split)))
        rgb_files = []
        dpt_files = []
        for split in splits:
            for scene in os.listdir(join(self.cfg.data_root, self.cfg.split, split)):
                rgb_files.extend(sorted(glob.glob(join(self.cfg.data_root, self.cfg.split, split, scene, 'photo', '*.jpg'))))
                dpt_files.extend(sorted(glob.glob(join(self.cfg.data_root, self.cfg.split, split, scene, 'depth', '*.png'))))
        frame_len = len(rgb_files)
        frame_sample = deepcopy(self.cfg.frames)
        frame_sample[1] =  frame_len if frame_sample[1] == -1 else frame_sample[1]
        # frames = frames[frame_sample[0]:frame_sample[1]:frame_sample[2]]
        self.rgb_files = rgb_files[frame_sample[0]:frame_sample[1]:frame_sample[2]]
        self.dpt_files = dpt_files[frame_sample[0]:frame_sample[1]:frame_sample[2]]
        self.metas = self.rgb_files
        Log.info(f'[Dataset]: {len(self.rgb_files)} frames in total.')
    
    def read_rgb(self, index):
        rgb_path = self.rgb_files[index]
        rgb = (np.asarray(imageio.imread(rgb_path)) / 255.).astype(np.float32)
        return rgb
    
    def read_dpt(self, index):
        dpt_path = self.dpt_files[index]
        dpt = (imageio.imread(dpt_path) / 1000.).astype(np.float32)
        if np.isnan(dpt).any(): dpt[np.isnan(dpt)] = dpt[np.isnan(dpt)==False].max()
        
        # ################
        # log depth
        # dpt = np.clip(dpt, 0.5, 80.)
        # dpt = (np.log(dpt) - np.log(0.5))/(np.log(80.) - np.log(0.5))
        # ################
        
        # ################
        # official depth
        if self.cfg.get('depth_norm', 'simp') == 'simp':
            dpt_min, dpt_max = np.percentile(dpt, 2.), np.percentile(dpt, 98.)
            if dpt_max - dpt_min < 1e-6: dpt_max = dpt_min + 2e-6
            dpt = np.clip(dpt, dpt_min, dpt_max)
            dpt = (dpt - dpt_min) / (dpt_max - dpt_min)
        elif self.cfg.get('depth_norm', 'simp') == 'log':
            dpt = np.clip(dpt, 0.5, 80.)
            dpt = (np.log(dpt) - np.log(0.5))/(np.log(80.) - np.log(0.5))
        elif self.cfg.get('depth_norm', 'simp') == 'disp':
            dpt = np.clip(dpt, 1., None)
            dpt = 1 / dpt
        else:
            import ipdb; ipdb.set_trace()
        # dpt = np.clip(dpt, 0.1, None)
        # dpt_min, dpt_max = np.percentile(dpt, 2.), np.percentile(dpt, 98.)
        # if dpt_max - dpt_min <= 1e-6: dpt_max = dpt_min + 2e-6
        # dpt = np.clip(dpt, dpt_min, dpt_max)
        # dpt = (dpt - dpt_min) / (dpt_max - dpt_min)
        # ################
        return dpt.astype(np.float32)
    
    def read_rgb_name(self, index: int) -> str:
        rgb_path = self.rgb_files[index]
        return 'splt' + rgb_path.split('/')[-3] + '_scene' + rgb_path.split('/')[-2] + '_frame' +  os.path.basename(rgb_path)
    
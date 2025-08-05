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

class Dataset(BaseDataset):
    def build_metas(self):
        splits = json.load(open(self.cfg.split_path))
        
        key = 'train_rgb_paths' if self.cfg.split == 'train' else 'test_rgb_paths'
        rgb_paths = splits[key]
        key = 'train_dpt_paths' if self.cfg.split == 'train' else 'test_dpt_paths'
        dpt_paths = splits[key]
        assert len(rgb_paths) == len(dpt_paths)
        
        frame_len = len(rgb_paths) 
        frame_sample = deepcopy(self.cfg.frames)
        frame_sample[1] =  frame_len if frame_sample[1] == -1 else frame_sample[1]
        rgb_paths = rgb_paths[frame_sample[0]:frame_sample[1]:frame_sample[2]]
        dpt_paths = dpt_paths[frame_sample[0]:frame_sample[1]:frame_sample[2]]
        
        self.rgb_files = [os.path.join(self.cfg.rgb_dir, rgb_path) for rgb_path in rgb_paths]
        self.dpt_files = [os.path.join(self.cfg.dpt_dir, dpt_path) for dpt_path in dpt_paths]
        self.metas = self.rgb_files
        Log.info(f'{self.cfg.split} split of HyperSim dataset: {len(self.rgb_files)} frames in total.')
    
    def read_rgb(self, index):
        rgb_path = self.rgb_files[index]
        rgb = (np.asarray(imageio.imread(rgb_path)) / 255.).astype(np.float32)
        return rgb
    
    def read_dpt(self, index):
        dpt_path = self.dpt_files[index]
        dpt = np.asarray(np.load(dpt_path)['data']).astype(np.float32)
        msk = ~np.isnan(dpt)
        if np.isnan(dpt).any(): dpt[np.isnan(dpt)] = dpt[np.isnan(dpt)==False].max()
        # try:
        #     if np.isnan(dpt).any(): dpt[np.isnan(dpt)] = dpt[np.isnan(dpt)==False].max()
        # except:
        #     import ipdb; ipdb.set_trace()
        return dpt.astype(np.float32), msk.astype(np.uint8)
    
    def read_rgb_name(self, index: int) -> str:
        # scene + camera + frame
        rgb_path = self.rgb_files[index]
        return rgb_path.split('/')[-4] + '_' + rgb_path.split('/')[-2] + '_' +  os.path.basename(rgb_path)
    
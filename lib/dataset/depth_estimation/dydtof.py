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
    
    def build_metas(self):
        self.dataset_name = 'dydtof'
        splits = json.load(open(self.cfg.split_path))
        data_root = self.cfg.data_root
        rgb_paths = splits['rgb_files']
        depth_paths = splits['depth_files']
        
        
        rgb_paths = [join(data_root, path) for path in tqdm(rgb_paths)]
        dpt_paths = [join(data_root, path) for path in tqdm(depth_paths)]
        
        frame_len = len(rgb_paths) 
        frame_sample = self.cfg.get('frames', [0, -1, 1])
        frame_sample[1] =  frame_len if frame_sample[1] == -1 else frame_sample[1]
        
        rgb_paths = rgb_paths[frame_sample[0]:frame_sample[1]:frame_sample[2]]
        dpt_paths = dpt_paths[frame_sample[0]:frame_sample[1]:frame_sample[2]]
        
        self.rgb_files = rgb_paths
        self.depth_files = dpt_paths
        
    def read_depth(self, index, depth=None):
        depth = np.load(self.depth_files[index])
        valid_mask = (depth != 50.).astype(np.uint8)  
        depth[~valid_mask] = depth[valid_mask].max()
        return depth, valid_mask
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
        self.dataset_name = 'hypersim'
        splits = json.load(open(self.cfg.split_path))
        key = 'train_rgb_paths' if self.cfg.split == 'train' else 'test_rgb_paths'
        rgb_paths = splits[key]
        key = 'train_dpt_paths' if self.cfg.split == 'train' else 'test_dpt_paths'
        dpt_paths = splits[key]
        assert len(rgb_paths) == len(dpt_paths)
        
        frame_len = len(rgb_paths) 
        frame_sample = self.cfg.get('frames', [0, -1, 1])
        frame_sample[1] =  frame_len if frame_sample[1] == -1 else frame_sample[1]
        
        rgb_paths = rgb_paths[frame_sample[0]:frame_sample[1]:frame_sample[2]]
        dpt_paths = dpt_paths[frame_sample[0]:frame_sample[1]:frame_sample[2]]
        if self.cfg.get('scene', None) is not None:
            scene = self.cfg.scene
            Log.info('Filtering scene: %s' % self.cfg.scene)
            rgb_paths = [rgb_path for rgb_path in rgb_paths if scene in rgb_path]
            dpt_paths = [dpt_path for dpt_path in dpt_paths if scene in dpt_path]
        
        assert len(rgb_paths) == len(dpt_paths)
        
        self.rgb_files = [os.path.join(self.cfg.rgb_dir, rgb_path) for rgb_path in rgb_paths]
        self.depth_files = [os.path.join(self.cfg.dpt_dir, dpt_path) for dpt_path in dpt_paths]
        
    def read_rgb_name(self, index):
        return '__'.join(self.rgb_files[index].split('/')[-4:])
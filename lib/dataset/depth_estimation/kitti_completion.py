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
        self.dataset_name = 'kitti_completion'
        splits = json.load(open(self.cfg.split_path))
        data_root = self.cfg.data_root
        rgb_paths = splits['rgb_files']
        depth_paths = splits['depth_files']
        low_depth_paths = splits['lowres_files']
        
        
        rgb_paths = [join(data_root, path) for path in tqdm(rgb_paths)]
        dpt_paths = [join(data_root, path) for path in tqdm(depth_paths)]
        low_dpt_paths = [join(data_root, path) for path in tqdm(low_depth_paths)]
        
        frame_len = len(rgb_paths) 
        frame_sample = self.cfg.get('frames', [0, -1, 1])
        frame_sample[1] =  frame_len if frame_sample[1] == -1 else frame_sample[1]
        
        rgb_paths = rgb_paths[frame_sample[0]:frame_sample[1]:frame_sample[2]]
        dpt_paths = dpt_paths[frame_sample[0]:frame_sample[1]:frame_sample[2]]
        low_dpt_paths = low_dpt_paths[frame_sample[0]:frame_sample[1]:frame_sample[2]]
        
        self.rgb_files = rgb_paths
        self.depth_files = dpt_paths
        self.low_files = low_dpt_paths
        
    def read_depth(self, index, depth=None):
        depth = cv2.imread(self.depth_files[index], cv2.IMREAD_ANYDEPTH)
        depth = np.asarray(depth / 256.).astype(np.float32)
        valid_mask = (depth != 0.).astype(np.uint8)
        # depth[~valid_mask] = depth[valid_mask].max()
        return depth, valid_mask
    
    def read_low_depth(self, file):
        depth = cv2.imread(file, cv2.IMREAD_ANYDEPTH)
        depth = np.asarray(depth / 256.).astype(np.float32)
        return depth
    
    def read_rgb_name(self, index):
        return '__'.join(self.rgb_files[index].split('/')[-4:])
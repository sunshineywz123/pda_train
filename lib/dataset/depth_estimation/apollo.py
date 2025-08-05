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
        self.dataset_name = 'apollo'
        self.split = self.cfg.split
        splits = json.load(open(self.cfg.split_path))
        data_root = self.cfg.data_root
        rgb_paths = splits['rgb_files']
        dpt_paths = splits['depth_files']
        low_dpt_paths = splits['depth_files']
        frame_len = len(rgb_paths) 
        frame_sample = self.cfg.get('frames', [0, -1, 1])
        frame_sample[1] =  frame_len if frame_sample[1] == -1 else frame_sample[1]
        
        rgb_paths = rgb_paths[frame_sample[0]:frame_sample[1]:frame_sample[2]]
        dpt_paths = dpt_paths[frame_sample[0]:frame_sample[1]:frame_sample[2]]
        low_dpt_paths = low_dpt_paths[frame_sample[0]:frame_sample[1]:frame_sample[2]]
        
        self.rgb_files = rgb_paths
        self.depth_files = dpt_paths
        # self.__DEPTH_C = np.array(1000.0 / (256 * 256 * 256 - 1), np.float32)
        
    def read_depth(self, index, depth=None):
        bgr = np.asarray(cv2.imread(self.depth_files[index]) / 255.)
        depth = ((bgr[..., 2] + bgr[..., 1] / 255.0) * 65536) / 100. 
        valid_mask = (depth < 80.)
        depth[~valid_mask] = 80.
        if self.split == 'train': return depth, np.ones_like(depth).astype(np.uint8)
        else: return depth, valid_mask.astype(np.uint8)
        # depth = cv2.imread(file)
        # depth = (depth[:, :, 0] * 256. * 256. + depth[:, :, 1] * 256. + depth[:, :, 2]) * self.__DEPTH_C
        # ill_mask = (depth > 80.)
        # depth[ill_mask] = 80.
        # return depth
        # import ipdb; ipdb.set_trace()
        # depth = cv2.imread(file)
        # depth = (depth[:, :, 0] * 256. * 256. + depth[:, :, 1] * 256. + depth[:, :, 2]) * self.__DEPTH_C
        
        # lidar_mask = (depth != 0.) & (depth < 80.)
        # gt_depth, gt_mask = self.read_depth(index)
        # # new_mask = gt_mask & lidar_mask
        # output_depth = np.zeros_like(gt_depth)
        # output_depth[lidar_mask] = gt_depth[lidar_mask]
        # depth = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        # depth = (depth[:, :, 0] * 256. * 256. + depth[:, :, 1] * 256. + depth[:, :, 2]) * self.__DEPTH_C
        # ixt = np.array([[640., 640.]])
        
        
        # return output_depth
    

# def generate_lidar_depth(depth, ixt, tar_lines = 64, reserve_ratio = 0.5):
    
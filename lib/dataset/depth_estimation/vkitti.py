import json
import os
import pickle
from copy import deepcopy
from glob import glob
from os.path import join

import cv2
import numpy as np
from torchvision.transforms import Compose
from tqdm import tqdm

from lib.dataset.depth_estimation.depth_estimation import \
    Dataset as BaseDataset
from lib.dataset.depth_estimation.depth_estimation import *
from lib.utils.pylogger import Log


class Dataset(BaseDataset):
    
    def build_metas(self):

        self.dataset_name = 'shift'
        self.split = self.cfg.split
        rgb_paths=[]
        dpt_paths=[]

        root_dir = os.path.join("/iag_ad_01/ad/yuanweizhong/datasets/vkitti")
        scenes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        for scene in scenes:
            situations = [s for s in os.listdir(os.path.join(root_dir, scene)) if os.path.isdir(os.path.join(root_dir, scene, s))]
            for situation in situations:
                for typ, paths in [('rgb', rgb_paths), ('depth', dpt_paths)]:
                    for cam in ['Camera_0', 'Camera_1']:
                        files = sorted(glob(os.path.join(root_dir, scene, situation, 'frames', typ, cam, '*.png')))
                        paths.extend(files)

        frame_len = len(rgb_paths) 
        frame_sample = self.cfg.get('frames', [0, -1, 1])
        frame_sample[1] =  frame_len if frame_sample[1] == -1 else frame_sample[1]

        # 按split划分数据
        split_idx = int(frame_len * 0.8)
        if self.split == 'train':
            rgb_paths = rgb_paths[:split_idx]
            dpt_paths = dpt_paths[:split_idx]
        else:
            rgb_paths = rgb_paths[split_idx:]
            dpt_paths = dpt_paths[split_idx:]
        
        self.rgb_files = rgb_paths
        self.depth_files = dpt_paths

    def read_rgb(self, index):
        rgb = cv2.imread(self.rgb_files[index], cv2.IMREAD_UNCHANGED)
        rgb = rgb.astype(np.float32)
        return rgb
    

    def read_depth(self, index,depth = None):

        depth = cv2.imread(self.depth_files[index], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        
        depth = depth.astype(np.float32)
        depth = depth / 100.0

        return depth, np.ones_like(depth).astype(np.uint8)

    
    def read_low_depth(self, file, index=None, **kwargs) -> np.ndarray:
        depth, _ = self.read_depth(index)

        target_h, target_w = depth.shape[:2]
        depth_lr = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        stride = 16
        anchors = []
        for y in range(0, target_h, stride):
            for x in range(0, target_w, stride):
                dy = np.random.randint(-2, 3)
                dx = np.random.randint(-2, 3)
                yy = np.clip(y + dy, 0, target_h - 1)
                xx = np.clip(x + dx, 0, target_w - 1)
                anchors.append((yy, xx))
        anchors = np.array(anchors)

        # 只在 anchor 点保留真实深度，其余为0
        sparse_depth = np.zeros((target_h, target_w), dtype=np.float32)
        sparse_depth[anchors[:,0], anchors[:,1]] = depth_lr[anchors[:,0], anchors[:,1]]

        return sparse_depth

import json
import os
import pickle
from copy import deepcopy
from os.path import join

import cv2
import numpy as np
from torchvision.transforms import Compose
from tqdm import tqdm

from lib.dataset.depth_estimation.depth_estimation import \
    Dataset as BaseDataset
from lib.dataset.depth_estimation.depth_estimation import *
from lib.utils.pylogger import Log
from PIL import Image

class Dataset(BaseDataset):
    
    def build_metas(self):

        self.dataset_name = 'shift'
        self.split = self.cfg.split
        # if self.split=='train':
        #     with open('/home/nas/users/pangbo/pl_htcode/senseguide/lightwheel_occ_infos_train.pkl','rb') as f: data_infos= pickle.load(f)
        # else:
        #     with open('/home/nas/users/pangbo/pl_htcode/senseguide/lightwheel_occ_infos_val.pkl','rb') as f: data_infos= pickle.load(f)

        with open ('/iag_ad_01/ad/yuanweizhong/huzeyu/pda_train/data/pl_htcode/processed_datasets/shift/train_split.json') as f :
            data_infos = json.load(f)

        rgb_paths=data_infos['rgb_files']
        dpt_paths=data_infos['depth_files']
        num = 5
        # num=-1
        for rgb_path,dpt_path in zip(rgb_paths[:num],dpt_paths[:num]):
            if os.path.exists(rgb_path) and os.path.exists(dpt_path):
                rgb_paths.append(rgb_path)
                dpt_paths.append(dpt_path)

        frame_len = len(rgb_paths) 
        frame_sample = self.cfg.get('frames', [0, -1, 1])
        frame_sample[1] =  frame_len if frame_sample[1] == -1 else frame_sample[1]
        
        self.rgb_files = rgb_paths
        self.depth_files = dpt_paths
        # self.low_files = low_dpt_paths
        # self.__DEPTH_C = np.array(1000.0 / (256 * 256 * 256 - 1), np.float32)

    def read_rgb(self, index):
        rgb = cv2.imread(self.rgb_files[index])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = np.asarray(rgb / 255.).astype(np.float32)
        return rgb

    def read_depth(self, index,depth = None):
        # depth_img = cv2.imread(self.depth_files[index], cv2.IMREAD_UNCHANGED)
        depth_img = Image.open(self.depth_files[index])
        depth_img = np.array(depth_img).astype(np.float64)
        depth = depth_img[:,:,0] + \
                (depth_img[:,:,1] * 256) + \
                (depth_img[:,:,2] * 256 * 256)
        # depth = depth /16777216.0
        depth = depth /16777216.0
        depth = depth * 1000.0

        # valid_mask = (depth < 80.)
        # depth[~valid_mask] = 80.
        return depth, np.ones_like(depth).astype(np.uint8)

    
    def read_low_depth(self, file,index=None):
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
        sparse_depth = np.zeros((target_h, target_w), dtype=depth_lr.dtype)
        sparse_depth[anchors[:,0], anchors[:,1]] = depth_lr[anchors[:,0], anchors[:,1]]

        # 可视化
        depth_vis = cv2.normalize(sparse_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite("decoded_depth_in_meters.png", depth_vis)
        return sparse_depth

# def generate_lidar_depth(depth, ixt, tar_lines = 64, reserve_ratio = 0.5):
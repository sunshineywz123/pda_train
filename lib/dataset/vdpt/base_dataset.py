import os
from abc import ABC, abstractmethod
from omegaconf import DictConfig
import cv2
import numpy as np
class Dataset(ABC):
    crop_size = 8
    window_size = 1
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self.cfg = DictConfig(kwargs)
        self.build_metas()
    
    @abstractmethod 
    def build_metas(self):
        pass
    
    @abstractmethod
    def read_rgbs(self, index):
        pass
    
    @abstractmethod
    def read_rgb_name(self, index):
        pass
    
    @abstractmethod
    def read_dpts(self, index):
        pass
    
    @staticmethod
    def get_metas_from_videos(video_len, video_start_idx, seq_len=8):
        frames = np.arange(video_len)
        frame_matrix = (np.asarray([np.arange(video_len) + i for i in range(seq_len*2 +1)]) - seq_len).T
        frame_matrix = np.clip(frame_matrix, 0, video_len-1, out=frame_matrix)
        metas = [(video_start_idx+frames[i], (frame_matrix[i] + video_start_idx).tolist()) for i in range(video_len)]
        return metas
    
    def __getitem__(self, index):
        rgb_name = self.read_rgb_name(index)
        rgbs, dpts = self.read_rgbs(index), self.read_dpts(index)
        
        assert(rgbs.shape[1:3] == dpts.shape[1:3]), "rgbs.shape: {}, dpts.shape: {}".format(rgbs.shape, dpts.shape)
        assert(len(rgbs.shape) == 4), "rgbs.shape: {}".format(rgbs.shape)
        assert(len(dpts.shape) == 3), "dpts.shape: {}".format(dpts.shape)
        
        h, w = rgbs.shape[1:3]
        crop_size = self.crop_size
        if h % crop_size != 0: rgbs = rgbs[:, (h%crop_size)//2:-((h%crop_size) - (h%crop_size)//2)]; dpts = dpts[:, (h%crop_size)//2:-((h%crop_size) - (h%crop_size)//2)]
        if w % crop_size != 0: rgbs = rgbs[:, :, (w%crop_size)//2:-((w%crop_size) - (w%crop_size)//2)]; dpts = dpts[:, :, (w%crop_size)//2:-((w%crop_size) - (w%crop_size)//2)]
        if self.cfg.get('resize_ratio', 1.) != 1.:
            rgbs = np.asarray([cv2.resize(rgb, (int(w*self.cfg.resize_ratio), int(h*self.cfg.resize_ratio)), interpolation=cv2.INTER_AREA) for rgb in rgbs])
            dpt = np.asarray([cv2.resize(dpt, (int(w*self.cfg.resize_ratio), int(h*self.cfg.resize_ratio)), interpolation=cv2.INTER_NEAREST) for dpt in dpts])
        return {'rgbs': rgbs.transpose(0, 3, 1, 2), 
                'dpts': dpts[:, None],
                'meta': {'rgb_name': rgb_name, 
                         'depth_norm': self.cfg.get('depth_norm', 'simp')}}
    def __len__(self):
        return len(self.metas)
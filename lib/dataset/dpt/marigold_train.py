import os
from abc import ABC, abstractmethod
from omegaconf import DictConfig
import cv2

class Dataset(ABC):
    crop_size = 8
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self.cfg = DictConfig(kwargs)
        self.build_metas()
    
    @abstractmethod 
    def build_metas(self):
        pass
    
    @abstractmethod
    def read_rgb(self, index):
        pass
    
    @abstractmethod
    def read_rgb_name(self, index):
        pass
    
    @abstractmethod
    def read_dpt(self, index):
        pass
    
    def __getitem__(self, index):
        rgb_name = self.read_rgb_name(index)
        rgb, dpt = self.read_rgb(index), self.read_dpt(index)
        
        assert(rgb.shape[:2] == dpt.shape[:2]), "rgb.shape: {}, dpt.shape: {}".format(rgb.shape, dpt.shape)
        assert(len(rgb.shape) == 3), "rgb.shape: {}".format(rgb.shape)
        assert(len(dpt.shape) == 2), "rgb.shape: {}".format(rgb.shape)
        
        h, w = rgb.shape[:2]
        crop_size = self.crop_size
        if h % crop_size != 0: rgb = rgb[(h%crop_size)//2:-((h%crop_size) - (h%crop_size)//2)]; dpt = dpt[(h%crop_size)//2:-((h%crop_size) - (h%crop_size)//2)]
        if w % crop_size != 0: rgb = rgb[:, (w%crop_size)//2:-((w%crop_size) - (w%crop_size)//2)]; dpt = dpt[:, (w%crop_size)//2:-((w%crop_size) - (w%crop_size)//2)]
        if self.cfg.get('resize_ratio', 1.) != 1.:
            rgb = cv2.resize(rgb, (int(w*self.cfg.resize_ratio), int(h*self.cfg.resize_ratio)), interpolation=cv2.INTER_AREA)
            dpt = cv2.resize(dpt, (int(w*self.cfg.resize_ratio), int(h*self.cfg.resize_ratio)), interpolation=cv2.INTER_NEAREST)
        return {'rgb': rgb.transpose(2, 0, 1), 
                'dpt': dpt[None],
                'meta': {'rgb_name': rgb_name, 
                         'depth_norm': self.cfg.get('depth_norm', 'simp')}}
    def __len__(self):
        return len(self.metas)
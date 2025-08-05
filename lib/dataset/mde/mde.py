import os
from abc import ABC, abstractmethod
from omegaconf import DictConfig
import cv2
import numpy as np

class Dataset(ABC):
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self.cfg = DictConfig(kwargs)
        self.build_metas()
        self.crop_size = self.cfg.get('crop_size', 8)
    
    @abstractmethod 
    def build_metas(self):
        pass
    
    @abstractmethod
    def read_rgb(self, index):
        # rgb: (h, w, 3)
        # range: [0., 1.]
        # type: np.ndarray float32
        pass
    
    @abstractmethod
    def read_rgb_name(self, index):
        pass
    
    @abstractmethod
    def read_dpt(self, index):
        pass
    
    def __getitem__(self, index):
        rgb_name = self.read_rgb_name(index)
        rgb, (dpt, msk) = self.read_rgb(index), self.read_dpt(index)
        
        # assert(rgb.shape[:2] == dpt.shape[:2]), "rgb.shape: {}, dpt.shape: {}".format(rgb.shape, dpt.shape)
        assert(len(rgb.shape) == 3), "rgb.shape: {}".format(rgb.shape)
        assert(len(dpt.shape) == 2), "dpt.shape: {}".format(dpt.shape)
        if msk is not None: assert(len(msk.shape) == 2), "msk.shape: {}".format(msk.shape)
        else: msk = np.ones_like(dpt).astype(np.uint8)
        
        h, w = rgb.shape[:2]
        crop_size = self.crop_size
        
        if self.cfg.get('resize_ratio', 1.) != 1.:
            rgb = cv2.resize(rgb, (int(w*self.cfg.resize_ratio), int(h*self.cfg.resize_ratio)), interpolation=cv2.INTER_AREA)
            dpt = cv2.resize(dpt, (int(w*self.cfg.resize_ratio), int(h*self.cfg.resize_ratio)), interpolation=cv2.INTER_NEAREST)
            msk = cv2.resize(msk, (int(w*self.cfg.resize_ratio), int(h*self.cfg.resize_ratio)), interpolation=cv2.INTER_NEAREST)
        
        h, w = rgb.shape[:2]
        if h % crop_size != 0: 
            rgb = rgb[(h%crop_size)//2:-((h%crop_size) - (h%crop_size)//2)]
            dpt = dpt[(h%crop_size)//2:-((h%crop_size) - (h%crop_size)//2)] 
            msk = msk[(h%crop_size)//2:-((h%crop_size) - (h%crop_size)//2)]
        if w % crop_size != 0: 
            rgb = rgb[:, (w%crop_size)//2:-((w%crop_size) - (w%crop_size)//2)] 
            dpt = dpt[:, (w%crop_size)//2:-((w%crop_size) - (w%crop_size)//2)]
            msk = msk[:, (w%crop_size)//2:-((w%crop_size) - (w%crop_size)//2)]
            
        return {'rgb': rgb.transpose(2, 0, 1), 
                'dpt': dpt[None],
                'msk': msk[None],
                'meta': {'rgb_name': rgb_name, 
                         'depth_norm': self.cfg.get('depth_norm', 'simp')}}
    def __len__(self):
        return len(self.metas)
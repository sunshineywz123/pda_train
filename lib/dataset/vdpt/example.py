import numpy as np
import os
import imageio
import cv2
from tqdm import tqdm
from os.path import join
import glob
from omegaconf import DictConfig
from lib.dataset.vdpt.base_dataset import Dataset as BaseDataset
class Dataset(BaseDataset):
    # def __init__(self, **kwargs):
        # super(Dataset, self).__init__()
        # self.cfg = DictConfig(kwargs)
        # self.data_root = self.cfg.data_root
    def build_metas(self):
        self.data_root = self.cfg.data_root
        image_files = glob.glob(os.path.join(self.data_root, '*.jpg')) + \
              glob.glob(os.path.join(self.data_root, '*.JPG')) + \
              glob.glob(os.path.join(self.data_root, '*.jpeg')) + \
              glob.glob(os.path.join(self.data_root, '*.png')) + \
              glob.glob(os.path.join(self.data_root, '*.PNG'))
        frame_len = self.cfg.get('frame_len', 1) 
        metas = BaseDataset.get_metas_from_videos(len(image_files), 0, seq_len=self.cfg.get('frame_len', 1))
        self.image_files = sorted([image_file for image_file in image_files if '.' != os.path.basename(image_file)[0]])
        self.max_image_size = self.cfg.get('max_image_size', 768)
        self.rgb_files = self.image_files
        # self.metas = metas[frame_len:-1:2*frame_len+1]
        self.metas = metas[frame_len:1000:frame_len]
        
    def __getitem__(self, index):
        rgb_path = self.rgb_files[self.metas[index][0]]
        rgbs = self.read_rgbs(index)
        h, w = rgbs.shape[1:3]
        crop_size = self.crop_size
        if h % crop_size != 0: rgbs = rgbs[:, (h%crop_size)//2:-((h%crop_size) - (h%crop_size)//2)]
        if w % crop_size != 0: rgbs = rgbs[:, :, (w%crop_size)//2:-((w%crop_size) - (w%crop_size)//2)]
        if self.cfg.get('resize_ratio', 1.) != 1.:
            rgbs = np.asarray([cv2.resize(rgb, (int(w*self.cfg.resize_ratio), int(h*self.cfg.resize_ratio)), interpolation=cv2.INTER_AREA) for rgb in rgbs])
        max_dim = max(rgbs.shape[1:3])
        ret = {}
        if max_dim != self.max_image_size and False:
            orig_size = rgbs.shape[1:3]
            h, w = rgbs.shape[1:3]
            scale = self.max_image_size / max_dim
            rgbs = np.asarray([cv2.resize(rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA) for rgb in rgbs])
            ret.update({'orig_size': orig_size})
        ret.update({'rgbs': rgbs.transpose(0, 3, 1, 2), 
                    'meta': {'rgb_names': ['{:04d}.jpg'.format(seq_id) for seq_id in self.metas[index][1]],
                             'ids': self.metas[index][1],
                             'window_idx': index}})
        return ret

    def read_rgb(self, index): 
        rgb_path = self.rgb_files[index]
        rgb = (np.asarray(imageio.imread(rgb_path)) / 255.).astype(np.float32)
        return rgb
    
    def read_rgbs(self, index):
        frame_idx, seq_ids = self.metas[index]
        rgbs = np.asarray([self.read_rgb(seq_id) for seq_id in seq_ids])
        return rgbs

    def __len__(self):
        return len(self.metas)
    
    def read_rgb_name(self, index):
        pass
    
    def read_dpts(self, index):
        pass
import numpy as np
import os
import imageio
import cv2
from tqdm import tqdm
from os.path import join
import glob
from omegaconf import DictConfig
class Dataset:
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self.cfg = DictConfig(kwargs)
        self.data_root = self.cfg.data_root
        image_files = glob.glob(os.path.join(self.data_root, '*.jpg')) + \
              glob.glob(os.path.join(self.data_root, '*.JPG')) + \
              glob.glob(os.path.join(self.data_root, '*.jpeg')) + \
              glob.glob(os.path.join(self.data_root, '*.png')) + \
              glob.glob(os.path.join(self.data_root, '*.PNG'))
        self.image_files = sorted([image_file for image_file in image_files if '.' != os.path.basename(image_file)[0]])
        self.image_files = self.image_files[:1000]
        self.max_image_size = self.cfg.get('max_image_size', 768)
        
    def __getitem__(self, index):
        img_path = self.image_files[index]
        img = np.asarray(imageio.imread(img_path))
        # resize
        max_dim = max(img.shape[:2])
        ret = {}
        if max_dim != self.max_image_size and False:
            orig_size = img.shape[:2]
            scale = self.max_image_size / max_dim
            img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
            ret.update({'orig_size': orig_size})
        
        # crop
        h, w = img.shape[:2]
        img = img[:h//8*8 if h%8!=0 else h, :w//8*8 if w%8!=0 else w]
        img = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)
        ret.update({'rgb': img, 'meta': {'rgb_name': os.path.basename(img_path), 'rgb_path': img_path}})
        return ret

    def __len__(self):
        return len(self.image_files)
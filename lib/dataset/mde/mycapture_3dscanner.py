import os
import numpy as np
from tqdm.auto import tqdm
from lib.dataset.mde.mde import Dataset as BaseDataset
from copy import deepcopy
import imageio
import glob
import cv2
import json
from lib.utils.pylogger import Log
import pandas as pd


class Dataset(BaseDataset):
    def build_metas(self):
        data_root = self.cfg.data_root
        self.rgb_files, self.dpt_files = [], []
        scenes = [self.cfg.scene]
        for scene in tqdm(scenes):
            rgb_files = sorted(glob.glob(os.path.join(
                data_root, scene, 'rgb', '*.jpg')))
            dpt_files = sorted(glob.glob(os.path.join(
                data_root, scene, 'depth', '*.png')))
            self.rgb_files += rgb_files
            self.dpt_files += dpt_files
        frames = self.cfg.frames
        s, e, i = frames[0], frames[1], frames[2]
        e = min(e, len(self.rgb_files)) if e != -1 else len(self.rgb_files)
        self.rgb_files = self.rgb_files[s:e:i]
        self.dpt_files = self.dpt_files[s:e:i]
        self.metas = self.rgb_files
        Log.info(f'Mycapture_3d_scanner {self.cfg.scene}: {len(self.rgb_files)} frames in total.')

    def read_rgb(self, index):
        rgb_path = self.rgb_files[index]
        rgb = (np.asarray(imageio.imread(rgb_path)) / 255.).astype(np.float32)
        return rgb

    def read_dpt(self, index):
        dpt_path = self.dpt_files[index]
        dpt = np.asarray(imageio.imread(dpt_path)) / 1000.
        return dpt.astype(np.float32), (dpt != 0).astype(np.uint8)

    def read_rgb_name(self, index: int) -> str:
        rgb_path = self.rgb_files[index]
        return os.path.basename(rgb_path)

    def __getitem__(self, index):
        ret_dict = super(Dataset, self).__getitem__(index)
        rgb = ret_dict['rgb'].transpose(1, 2, 0)
        conf = (imageio.imread(self.dpt_files[index].replace('depth', 'conf'))).astype(np.uint8)

        tar_w = self.cfg.get('tar_w', 640)  # 640, 768, 1024, 1920
        assert (tar_w % 4 == 0)
        tar_h = int(tar_w / 4 * 3)
        if rgb.shape[0] != tar_h:
            rgb = cv2.resize(rgb, (tar_w, tar_h),
                             interpolation=cv2.INTER_AREA).astype(np.float32)
            
        
        ret_dict.update({
            'rgb': rgb.transpose(2, 0, 1),
            'lowres_dpt': ret_dict['dpt'],
            'conf': conf[None],
        })
        return ret_dict
    
    @staticmethod
    def rotate_image(img, direction):
        if direction == 'Up':
            pass
        elif direction == 'Left':
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif direction == 'Right':
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif direction == 'Down':
            img = cv2.rotate(img, cv2.ROTATE_180)
        else:
            raise Exception(f'No such direction (={direction}) rotation')
        return img

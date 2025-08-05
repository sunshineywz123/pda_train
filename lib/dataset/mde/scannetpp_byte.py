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
from trdparties.colmap.read_write_model import read_images_text
from os.path import join


class Dataset(BaseDataset):
    def build_metas(self):
        
        self.root_dir = join(self.cfg.data_root, 'data', self.cfg.scene, 'iphone')
        
        images = read_images_text(join(self.root_dir, 'colmap/images.txt'))
        
        img_names = [] 
        for id in images:
            img_names.append(images[id].name)
        img_names = sorted(img_names)
        
        self.rgb_files, self.dpt_files, self.low_files = [], [], []
        for img_name in tqdm(img_names):
            self.rgb_files.append(join(self.root_dir, 'rgb', img_name))
            self.dpt_files.append(join(self.root_dir, 'render_depth', img_name[:-4] + '.png'))
            self.low_files.append(join(self.root_dir, 'depth', img_name[:-4] + '.png'))
        frames = self.cfg.frames
        s, e, i = frames[0], frames[1], frames[2]
        e = min(e, len(self.rgb_files)) if e != -1 else len(self.rgb_files)
        self.rgb_files = self.rgb_files[s:e:i]
        self.dpt_files = self.dpt_files[s:e:i]
        self.low_files = self.low_files[s:e:i]
        self.metas = self.rgb_files
        Log.info(f'Scannetpp {self.cfg.scene}: {len(self.rgb_files)} frames in total.')

    def read_rgb(self, index):
        rgb_path = self.rgb_files[index]
        rgb = (np.asarray(imageio.imread(rgb_path)) / 255.).astype(np.float32)
        return rgb

    def read_dpt(self, index):
        dpt_path = self.dpt_files[index]
        dpt = np.asarray(imageio.imread(dpt_path)) / 1000.
        return dpt.astype(np.float32), (dpt != 0).astype(np.uint8)

    def read_rgb_name(self, index: int) -> str:
        # scene + camera + frame
        rgb_path = self.rgb_files[index]
        return rgb_path.split('/')[-3] + '_' + os.path.basename(rgb_path)

    def __getitem__(self, index):
        ret_dict = super(Dataset, self).__getitem__(index)
        rgb = ret_dict['rgb'].transpose(1, 2, 0)
        dpt = ret_dict['dpt'][0]
        lowres_dpt = (imageio.imread(
            self.low_files[index]) / 1000.).astype(np.float32)

        tar_w = self.cfg.get('tar_w', 640)  # 640, 768, 1024, 1920
        assert (tar_w % 4 == 0)
        tar_h = int(tar_w / 4 * 3)
        if rgb.shape[0] != tar_h:
            rgb = cv2.resize(rgb, (tar_w, tar_h),
                             interpolation=cv2.INTER_AREA).astype(np.float32)
            dpt = cv2.resize(dpt, (tar_w, tar_h),
                             interpolation=cv2.INTER_NEAREST).astype(np.float32)
        msk = dpt != 0.
        ret_dict.update({
            'rgb': rgb.transpose(2, 0, 1),
            'dpt': dpt[None],
            'lowres_dpt': lowres_dpt[None],
            'msk': msk[None].astype(np.uint8)
        })
        return ret_dict

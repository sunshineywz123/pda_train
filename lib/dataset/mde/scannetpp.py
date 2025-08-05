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
from trdparties.colmap.read_write_model import read_model
from os.path import join

class Dataset(BaseDataset):
    def build_metas(self):
        scene_dir = join(self.cfg.data_root, self.cfg.scene_id)
        cams, images, points = read_model(join(scene_dir, 'iphone/colmap')) # type: ignore
        
        self.rgb_files, self.dpt_files, self.low_files = [], [], []
        for k in images:
            image = images[k]
            rgb_path = join(scene_dir, 'iphone/rgb', image.name)
            dpt_path = join(scene_dir, 'iphone/render_depth', image.name[:-4] + '.png')
            low_path = join(scene_dir, 'iphone/depth', image.name[:-4] + '.png')
            self.rgb_files.append(rgb_path)
            self.dpt_files.append(dpt_path)
            self.low_files.append(low_path)
        self.metas = self.rgb_files
        Log.info(f'[Dataset]: {len(self.rgb_files)} frames in total.')
    
    def read_rgb(self, index):
        rgb_path = self.rgb_files[index]
        rgb = (np.asarray(imageio.imread(rgb_path)) / 255.).astype(np.float32)
        return rgb

    
    def read_dpt(self, index):
        dpt_path = self.dpt_files[index]
        dpt = np.asarray(imageio.imread(dpt_path)) / 1000.
        if self.cfg.get('depth_norm', 'simp') == 'simp':
            dpt_min, dpt_max = np.percentile(dpt, 2.), np.percentile(dpt, 98.)
            if dpt_max - dpt_min < 1e-6: dpt_max = dpt_min + 2e-6
            dpt = np.clip(dpt, dpt_min, dpt_max)
            dpt = (dpt - dpt_min) / (dpt_max - dpt_min)
        elif self.cfg.get('depth_norm', 'simp') == 'log':
            dpt = np.clip(dpt, 0.5, 80.)
            dpt = (np.log(dpt) - np.log(0.5))/(np.log(80.) - np.log(0.5))
        elif self.cfg.get('depth_norm', 'simp') == 'disp':
            dpt = np.clip(dpt, 1., None)
            dpt = 1 / dpt
        elif self.cfg.get('depth_norm', 'nonorm') == 'nonorm':
            pass
        else:
            import ipdb; ipdb.set_trace()
        return dpt.astype(np.float32), (dpt != 0).astype(np.uint8)
    
    def read_rgb_name(self, index: int) -> str:
        # scene + camera + frame
        rgb_path = self.rgb_files[index]
        return rgb_path.split('/')[-4] + '_' +  os.path.basename(rgb_path)
    
    def __getitem__(self, index):
        ret_dict = super(Dataset, self).__getitem__(index)
        rgb = ret_dict['rgb'].transpose(1, 2, 0)
        dpt = ret_dict['dpt'][0]
        lowres_dpt = (imageio.imread(self.low_files[index]) / 1000.).astype(np.float32)
        
        rgb = cv2.resize(rgb, (1024, 768), interpolation=cv2.INTER_AREA).astype(np.float32)
        dpt = cv2.resize(dpt, (1024, 768), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        msk = dpt!=0.
        
        ret_dict.update({
            'rgb': rgb.transpose(2, 0, 1),
            'dpt': dpt[None],
            'lowres_dpt': lowres_dpt[None],
            'msk': msk[None].astype(np.uint8)
        })
        return ret_dict
    
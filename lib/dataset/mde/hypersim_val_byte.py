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

class Dataset(BaseDataset):
    def build_metas(self):
        scene = self.cfg.scene
        cam = self.cfg.cam
        rgb_dir = self.cfg.rgb_dir
        dpt_dir = self.cfg.dpt_dir
        rgb_paths = sorted(glob.glob(os.path.join(rgb_dir, 'all', scene, 'images', 'scene_' + cam + '_final_preview', '*.tonemap.jpg')))
        dpt_paths = sorted(glob.glob(os.path.join(dpt_dir, 'all', scene, 'images', 'scene_' + cam + '_geometry_hdf5', '*.depth.npz')))
        self.rgb_files = rgb_paths
        self.dpt_files = dpt_paths
        self.metas = self.rgb_files
        Log.info(f'{self.cfg.split} split of HyperSim dataset: {len(self.rgb_files)} frames in total.')
    
    def read_rgb(self, index):
        rgb_path = self.rgb_files[index]
        rgb = (np.asarray(imageio.imread(rgb_path)) / 255.).astype(np.float32)
        return rgb
    
    def read_dpt(self, index):
        dpt_path = self.dpt_files[index]
        dpt = np.asarray(np.load(dpt_path)['data']).astype(np.float32)
        msk = ~np.isnan(dpt)
        if np.isnan(dpt).any(): dpt[np.isnan(dpt)] = dpt[np.isnan(dpt)==False].max()
        return dpt.astype(np.float32), msk.astype(np.uint8)
    
    def read_rgb_name(self, index: int) -> str:
        rgb_path = self.rgb_files[index]
        return rgb_path.split('/')[-4] + '_' + rgb_path.split('/')[-2] + '_' +  os.path.basename(rgb_path)
    
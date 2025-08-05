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
        scenes = sorted(os.listdir(self.cfg.data_root))
        if self.cfg.split == 'train':
            frames = json.load(open(os.path.join(self.cfg.data_root, 'ai_001_001/train_frames.json')))
        else:
            frames = json.load(open(os.path.join(self.cfg.data_root, 'ai_001_001/test_frames.json')))
        frame_len = len(frames)
        frame_sample = deepcopy(self.cfg.frames)
        frame_sample[1] =  frame_len if frame_sample[1] == -1 else frame_sample[1]
        frames = frames[frame_sample[0]:frame_sample[1]:frame_sample[2]]
        self.rgb_files = [os.path.join(self.cfg.data_root, frame.replace('geometry_hdf5', 'final_preview').replace('.depth_meters_plane.npz', '.color.jpg')) for frame in frames]
        self.dpt_files = [os.path.join(self.cfg.data_root, frame) for frame in frames]
        self.metas = self.rgb_files
        Log.info(f'[Dataset]: {len(self.rgb_files)} frames in total.')
    
    def read_rgb(self, index):
        rgb_path = self.rgb_files[index]
        rgb = (np.asarray(imageio.imread(rgb_path)) / 255.).astype(np.float32)
        return rgb
    
    def read_dpt(self, index):
        dpt_path = self.dpt_files[index]
        dpt = np.asarray(np.load(dpt_path)['data']).astype(np.float32)
        msk = ~np.isnan(dpt)
        if np.isnan(dpt).any(): dpt[np.isnan(dpt)] = dpt[np.isnan(dpt)==False].max()
        
        # official depth
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
        return dpt.astype(np.float32), msk.astype(np.uint8)
    
    def read_rgb_name(self, index: int) -> str:
        # scene + camera + frame
        rgb_path = self.rgb_files[index]
        return rgb_path.split('/')[-4] + '_' + rgb_path.split('/')[-2] + '_' +  os.path.basename(rgb_path)
    
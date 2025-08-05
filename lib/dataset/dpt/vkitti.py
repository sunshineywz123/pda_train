import os
import numpy as np
from tqdm.auto import tqdm
from lib.dataset.dpt.marigold_train import Dataset as BaseDataset
from copy import deepcopy
import imageio
from lib.utils.pylogger import Log


class Dataset(BaseDataset):
    def build_metas(self):
        tar_dirs = []
        for scene in self.cfg.scenes:
            for seq in self.cfg.sequences:
                tar_dirs.append(os.path.join(self.cfg.data_root, scene, seq))
        self.rgb_files = []
        self.dpt_files = []
        for tar_dir in tqdm(tar_dirs, desc='Scanning directories of vkitti'):
            frames = sorted(os.listdir(os.path.join(tar_dir, 'frames', 'rgb', 'Camera_0')))
            frame_len = len(frames)
            frame_sample = deepcopy(self.cfg.frames)
            frame_sample[1] =  frame_len if frame_sample[1] == -1 else frame_sample[1]
            frames = frames[frame_sample[0]:frame_sample[1]:frame_sample[2]]
            self.rgb_files += [os.path.join(tar_dir, 'frames', 'rgb', 'Camera_0', frame) for frame in frames]
            self.rgb_files += [os.path.join(tar_dir, 'frames', 'rgb', 'Camera_1', frame) for frame in frames]
            self.dpt_files += [os.path.join(tar_dir, 'frames', 'depth', 'Camera_0', frame.replace('rgb', 'depth').replace('.jpg', '.png')) for frame in frames]
            self.dpt_files += [os.path.join(tar_dir, 'frames', 'depth', 'Camera_1', frame.replace('rgb', 'depth').replace('.jpg', '.png')) for frame in frames]
        Log.info(f'[Dataset]: {len(self.rgb_files)} frames in total.')
        self.metas = self.rgb_files
    
    def read_rgb(self, index):
        rgb_path = self.rgb_files[index]
        rgb = (np.asarray(imageio.imread(rgb_path)) / 255.).astype(np.float32)
        return rgb
    
    def read_dpt(self, index):
        dpt_path = self.dpt_files[index]
        dpt = (np.asarray(imageio.imread(dpt_path)) / 100.).astype(np.float32)
        
        # ################
        # log depth
        # dpt = np.clip(dpt, 0.5, 80.)
        # dpt = (np.log(dpt) - np.log(0.5))/(np.log(80.) - np.log(0.5))
        # ################
        
        
        # ################
        # official depth
        if self.cfg.get('depth_norm', 'simp') == 'simp':
            dpt = np.clip(dpt, None, 80.)
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
        else:
            import ipdb; ipdb.set_trace()
        # ################
        # dpt = dpt / 80.
        # dpt = (np.log(dpt) - np.log(0.5))/(np.log(80.) - np.log(0.5))
        # dpt_min, dpt_max = np.percentile(dpt, 2.), np.percentile(dpt, 98.)
        # dpt = (dpt - dpt_min) / (dpt_max - dpt_min)
        return dpt
    
    def read_rgb_name(self, index: int) -> str:
        rgb_path = self.rgb_files[index]
        # scene + weather + camera + frame_name
        return rgb_path.split('/')[-6] + '_' + rgb_path.split('/')[-5] + '_' + rgb_path.split('/')[-2] + '_' +  os.path.basename(rgb_path)
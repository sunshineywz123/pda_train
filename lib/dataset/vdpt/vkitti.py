import os
import numpy as np
from tqdm.auto import tqdm
from lib.dataset.vdpt.base_dataset import Dataset as BaseDataset
from copy import deepcopy
import imageio
from lib.utils.pylogger import Log


class Dataset(BaseDataset):
    def build_metas(self):
        tar_dirs = []
        for scene in self.cfg.scenes:
            for seq in self.cfg.sequences:
                tar_dirs.append(os.path.join(self.cfg.data_root, scene, seq))
        rgb_files = []
        dpt_files = []
        global_metas = []
        for tar_dir in tqdm(tar_dirs, desc='Scanning directories of vkitti'):
            frames = sorted(os.listdir(os.path.join(tar_dir, 'frames', 'rgb', 'Camera_0')))
            # frame_len = len(frames)
            # frame_sample = deepcopy(self.cfg.frames)
            # frame_sample[1] =  frame_len if frame_sample[1] == -1 else frame_sample[1]
            # frames = frames[frame_sample[0]:frame_sample[1]:frame_sample[2]]
            
            rgb_video = [os.path.join(tar_dir, 'frames', 'rgb', 'Camera_0', frame) for frame in frames]
            dpt_video = [os.path.join(tar_dir, 'frames', 'depth', 'Camera_0', frame.replace('rgb', 'depth').replace('.jpg', '.png')) for frame in frames]
            metas = self.get_metas_from_videos(len(rgb_video), len(rgb_files), self.cfg.get('frame_len', 8))
            rgb_files.extend(rgb_video)
            dpt_files.extend(dpt_video)
            global_metas.extend(metas)
            
            
            frames = sorted(os.listdir(os.path.join(tar_dir, 'frames', 'rgb', 'Camera_1')))
            rgb_video = [os.path.join(tar_dir, 'frames', 'rgb', 'Camera_1', frame) for frame in frames]
            dpt_video = [os.path.join(tar_dir, 'frames', 'depth', 'Camera_1', frame.replace('rgb', 'depth').replace('.jpg', '.png')) for frame in frames]
            metas = self.get_metas_from_videos(len(rgb_video), len(rgb_files), self.cfg.get('frame_len', 8))
            rgb_files.extend(rgb_video)
            dpt_files.extend(dpt_video)
            global_metas.extend(metas)
        assert(len(rgb_files) == len(dpt_files)), f'{len(rgb_files)} != {len(dpt_files)}'
        frame_len = len(global_metas)
        frame_sample = deepcopy(self.cfg.frames)
        frame_sample[1] =  frame_len if frame_sample[1] == -1 else frame_sample[1]
        # rgb_files = rgb_files[frame_sample[0]:frame_sample[1]:frame_sample[2]]
        # dpt_files = dpt_files[frame_sample[0]:frame_sample[1]:frame_sample[2]]
        global_metas = global_metas[frame_sample[0]:frame_sample[1]:frame_sample[2]]
        self.rgb_files = rgb_files
        self.dpt_files = dpt_files
        self.metas = global_metas
        Log.info(f'[Dataset]: {len(self.metas)} frames in total.')
    
    def read_rgb(self, index):
        rgb_path = self.rgb_files[index]
        rgb = (np.asarray(imageio.imread(rgb_path)) / 255.).astype(np.float32)
        return rgb
    
    def read_rgbs(self, index):
        frame_idx, seq_ids = self.metas[index]
        rgbs = np.asarray([self.read_rgb(seq_id) for seq_id in seq_ids])
        return rgbs
    
    def read_dpts(self, index):
        frame_idx, seq_ids = self.metas[index]
        dpts = np.asarray([self.read_dpt(seq_id) for seq_id in seq_ids])
        return dpts
    
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
        frame_idx, seq_ids = self.metas[index]
        rgb_path = self.rgb_files[frame_idx]
        # scene + weather + camera + frame_name
        return rgb_path.split('/')[-6] + '_' + rgb_path.split('/')[-5] + '_' + rgb_path.split('/')[-2] + '_' +  os.path.basename(rgb_path)
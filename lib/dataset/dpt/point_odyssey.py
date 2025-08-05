import os
import numpy as np
from tqdm.auto import tqdm
from lib.dataset.dpt.marigold_train import Dataset as BaseDataset
from copy import deepcopy
import imageio
import glob
import cv2
import json
from lib.utils.pylogger import Log

class Dataset(BaseDataset):
    def build_metas(self):
        root_dir = '/'.join(self.cfg.data_root.split('/')[:-1])
        if os.path.exists(os.path.join(root_dir, 'rgb_files.json')) \
            and os.path.exists(os.path.join(root_dir, 'dpt_files.json')):
            rgb_files = json.load(open(os.path.join(root_dir, 'rgb_files.json'), 'r'))
            dpt_files = json.load(open(os.path.join(root_dir, 'dpt_files.json'), 'r'))
        else:
            scenes = sorted(os.listdir(self.cfg.data_root))
            scenes = sorted([scene for scene in scenes if 'mp4' not in scene])
            rgb_files = []
            dpt_files = []
            for scene in tqdm(scenes):
                rgb_files += sorted(glob.glob(os.path.join(self.cfg.data_root, scene, 'rgbs', '*')))
                dpt_files += sorted(glob.glob(os.path.join(self.cfg.data_root, scene, 'depths', '*')))
            try:
                json_data = json.dumps(rgb_files, ensure_ascii=False, indent=4)
                with open(os.path.join(root_dir, 'rgb_files.json'), 'w', encoding='utf-8') as file:
                    file.write(json_data)
                json_data = json.dumps(dpt_files, ensure_ascii=False, indent=4)
                with open(os.path.join(root_dir, 'dpt_files.json'), 'w', encoding='utf-8') as file:
                    file.write(json_data)
            except:
                Log.warn('Failed to save json files.')
                pass
        assert(len(rgb_files) == len(dpt_files)), f'{len(rgb_files)} != {len(dpt_files)}'
        frame_len = len(rgb_files)
        frame_sample = deepcopy(self.cfg.frames)
        frame_sample[1] =  frame_len if frame_sample[1] == -1 else frame_sample[1]
        rgb_files = rgb_files[frame_sample[0]:frame_sample[1]:frame_sample[2]]
        dpt_files = dpt_files[frame_sample[0]:frame_sample[1]:frame_sample[2]]
        self.rgb_files = rgb_files
        self.dpt_files = dpt_files
        self.metas = self.rgb_files
        Log.info(f'[Dataset]: {len(self.rgb_files)} frames in total.')
    
    def read_rgb(self, index):
        rgb_path = self.rgb_files[index]
        rgb = (np.asarray(imageio.imread(rgb_path)) / 255.).astype(np.float32)
        rgb = rgb[30:-30, 160:-160]
        return rgb
    
    def read_dpt(self, index):
        dpt_path = self.dpt_files[index]
        dpt = np.asarray(imageio.imread(dpt_path)).astype(np.float32) / 1000.
        dpt = dpt[30:-30, 160:-160]
        if (dpt==0.).any(): dpt[dpt==0.] = dpt[dpt!=0].max()
        # ################
        # log depth
        # dpt = np.clip(dpt, 0.5, 80.)
        # dpt = (np.log(dpt) - np.log(0.5))/(np.log(80.) - np.log(0.5))
        # ################
        
        # ################
        # official depth
        if self.cfg.get('depth_norm', 'simp') == 'simp':
            dpt = np.clip(dpt, 0.5, 80.)
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
        # dpt = np.clip(dpt, 0.1, None)
        # dpt_min, dpt_max = np.percentile(dpt, 2.), np.percentile(dpt, 98.)
        # if dpt_max - dpt_min <= 1e-6: dpt_max = dpt_min + 2e-6
        # dpt = np.clip(dpt, dpt_min, dpt_max)
        # dpt = (dpt - dpt_min) / (dpt_max - dpt_min)
        # ################
        return dpt.astype(np.float32)
    
    def read_rgb_name(self, index: int) -> str:
        rgb_path = self.rgb_files[index]
        return rgb_path.split('/')[-2] + '_' +  os.path.basename(rgb_path)
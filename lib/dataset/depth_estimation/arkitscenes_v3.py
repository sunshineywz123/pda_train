from lib.dataset.depth_estimation.depth_estimation import Dataset as BaseDataset
from lib.dataset.depth_estimation.depth_estimation import *
from os.path import join
import os
from torchvision.transforms import Compose
import pandas as pd
from tqdm import tqdm
import glob

class Dataset(BaseDataset):
    def build_metas(self):
        if hasattr(self.cfg.transforms[0], 'width'):
            self.dataset_name = 'arkitscenes_{}'.format(self.cfg.transforms[0].width)
        else:
            self.dataset_name = 'arkitscenes'
        meta_file = pd.read_csv(self.cfg.meta_file)
        self.scene_direction = {str(meta_file['video_id'][idx]): meta_file['sky_direction'][idx] for idx in range(len(meta_file['video_id']))}
        split = 'Training' if self.cfg.split == 'train' else 'Validation'
        scenes = sorted(os.listdir(self.cfg.data_root + '/' + split))
        if 'DEBUG' in os.environ and os.environ['DEBUG'] == 'HTCODE': scenes = scenes[:20]
        data_root = self.cfg.data_root + '/' + split
        self.rgb_files, self.depth_files, self.low_files, self.conf_files = [], [], [], []
        split_file_lines = open(self.cfg.split_file, 'r').readlines()
        for line in tqdm(split_file_lines, desc='Loading data: arkitscenes'):
            rgb_file, depth_file, low_file = line.strip().split(' ')
            self.rgb_files.append(join(data_root, rgb_file))
            self.depth_files.append(join(data_root, depth_file))
            self.low_files.append(join(data_root, low_file))
            self.conf_files.append(join(data_root, low_file.replace('lowres_depth', 'confidence')))
        frames = self.cfg.get('frames', [0, -1, 1])
        s, e, i = frames[0], frames[1], frames[2]
        e = min(e, len(self.rgb_files)) if e != -1 else len(self.rgb_files)
        self.rgb_files = self.rgb_files[s:e:i]
        self.depth_files = self.depth_files[s:e:i]
        self.low_files = self.low_files[s:e:i]
        self.conf_files = self.conf_files[s:e:i]

    def read_rgb_name(self, index):
        return '{}_{}'.format(self.dataset_name, self.rgb_files[index].split('/')[-1])
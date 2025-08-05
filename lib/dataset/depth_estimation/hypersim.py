from lib.dataset.depth_estimation.depth_estimation import Dataset as BaseDataset
from lib.dataset.depth_estimation.depth_estimation import *
from os.path import join
import os
from torchvision.transforms import Compose
import json
from copy import deepcopy
from lib.utils.pylogger import Log
from tqdm import tqdm

class Dataset(BaseDataset):
    def build_transforms(self):
        self.transform = Compose([
            # Resize(
            #     width=None,
            #     height=None,
            #     resize_target=True if self.cfg.split == 'train' else False,
            #     keep_aspect_ratio=True,
            #     ensure_multiple_of=14,
            #     resize_method='lower_bound',
            #     image_interpolation_method=cv2.INTER_AREA,
            # ),
            # NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
    def read_rgb_name(self, index):
        return '__'.join(self.rgb_files[index].split('/')[-4:])
    
    def build_metas(self):
        self.dataset_name = 'hypersim'
        splits = json.load(open(self.cfg.split_path))
        
        key = 'train_rgb_paths' if self.cfg.split == 'train' else 'test_rgb_paths'
        # key = 'test_rgb_paths'
        rgb_paths = splits[key]
        key = 'train_dpt_paths' if self.cfg.split == 'train' else 'test_dpt_paths'
        # key = 'test_dpt_paths'
        dpt_paths = splits[key]
        assert len(rgb_paths) == len(dpt_paths)
        
        frame_len = len(rgb_paths) 
        frame_sample = deepcopy(self.cfg.frames)
        frame_sample[1] =  frame_len if frame_sample[1] == -1 else frame_sample[1]
        rgb_paths = rgb_paths[frame_sample[0]:frame_sample[1]:frame_sample[2]]
        dpt_paths = dpt_paths[frame_sample[0]:frame_sample[1]:frame_sample[2]]
        
        if self.cfg.get('scene', None) is not None:
            scene = self.cfg.scene
            Log.info(f'Filtering frames with scene: {scene}')
            rgb_paths = [rgb_path for rgb_path in rgb_paths if (scene in rgb_path and 'cam_00' in rgb_path)]
            dpt_paths = [dpt_path for dpt_path in dpt_paths if (scene in dpt_path and 'cam_00' in dpt_path)]
        
        self.rgb_files = [os.path.join(self.cfg.rgb_dir, rgb_path) for rgb_path in rgb_paths]
        self.depth_files = [os.path.join(self.cfg.dpt_dir, dpt_path) for dpt_path in dpt_paths]
        self.metas = self.rgb_files
        Log.info(f'{self.cfg.split} split of HyperSim dataset: {len(self.rgb_files)} frames in total.')
        
        # data_root = self.cfg.data_root
        # meta_files = [line.strip() for line in open(join(data_root, self.cfg.split_path), 'r').readlines()]
        # self.rgb_files = []
        # self.depth_files = []
        # for meta_file in meta_files:
        #     rgb_file, depth_file, _ = meta_file.split(' ')
        #     self.rgb_files.append(join(data_root, rgb_file))
        #     self.depth_files.append(join(data_root, depth_file))
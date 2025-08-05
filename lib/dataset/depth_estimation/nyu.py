from lib.dataset.depth_estimation.depth_estimation import Dataset as BaseDataset
from lib.dataset.depth_estimation.depth_estimation import *
from os.path import join
import os
from torchvision.transforms import Compose

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
    
    def build_metas(self):
        self.dataset_name = 'nyuv2'
        data_root = self.cfg.data_root
        meta_files = [line.strip() for line in open(join(data_root, self.cfg.split_path), 'r').readlines()]
        self.rgb_files = []
        self.depth_files = []
        for meta_file in meta_files:
            rgb_file, depth_file, _ = meta_file.split(' ')
            self.rgb_files.append(join(data_root, rgb_file))
            self.depth_files.append(join(data_root, depth_file))
            
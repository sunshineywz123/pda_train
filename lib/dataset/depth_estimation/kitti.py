from lib.dataset.depth_estimation.depth_estimation import Dataset as BaseDataset
from lib.dataset.depth_estimation.depth_estimation import *
from os.path import join
import os
from torchvision.transforms import Compose

class Dataset(BaseDataset):
    def build_transforms(self):
        self.transform = Compose([
            # Resize(
            #     width=518,
            #     height=518,
            #     resize_target=True if self.cfg.split == 'train' else False,
            #     keep_aspect_ratio=True,
            #     ensure_multiple_of=8,
            #     resize_method='lower_bound',
            #     image_interpolation_method=cv2.INTER_CUBIC,
            # ),
            # NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
    
    def build_metas(self):
        self.dataset_name = 'kitti'
        data_root = self.cfg.data_root
        meta_files = [line.strip() for line in open(self.cfg.split_path, 'r').readlines()]
        self.rgb_files = []
        self.depth_files = []
        for meta_file in meta_files:
            rgb_file, depth_file = meta_file.split(' ')
            self.rgb_files.append(join(data_root, rgb_file[1:]))
            self.depth_files.append(join(data_root, depth_file[1:]))
            
    def read_depth(self, index):
        depth = imageio.imread(self.depth_files[index]) / 256.
        return super().read_depth(index, depth)
    
    def read_rgb_name(self, index):
        return '__'.join(self.rgb_files[index].split('/')[-4:])
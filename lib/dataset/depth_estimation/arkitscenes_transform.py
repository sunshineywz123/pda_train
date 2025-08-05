from lib.dataset.depth_estimation.depth_estimation import *
from os.path import join
import os
from torchvision.transforms import Compose
import pandas as pd
from tqdm import tqdm
import glob
from lib.dataset.depth_estimation.arkitscenes import Dataset as BaseDataset

class Dataset(BaseDataset):
    def build_transforms(self):
        if self.cfg.split == 'train':
            train_crop_size = self.cfg.get('train_crop_size', 630)
            down_scale = self.cfg.get('down_scale', 7.5)
            width = self.cfg.get('width', None)
            height = self.cfg.get('height', None)
            resize_target = self.cfg.get('resize_target', True)
            Log.info(f"Using train width {width}")
            Log.info(f"Using train height {height}")
            Log.info(f"Using crop size {train_crop_size}")
            Log.info(f"Using down scale {down_scale}")
            Log.info(f"Resize target: {resize_target}")
            self.transform = Compose([
                                    Resize(width=width,
                                             height=height,
                                             resize_target=resize_target,
                                             keep_aspect_ratio=True,
                                             ensure_multiple_of=1,
                                             resize_method='lower_bound',
                                             image_interpolation_method=cv2.INTER_AREA),
                                    PrepareForNet(), 
                                    Crop(train_crop_size, down_scale=down_scale)
                                      ])
        else:
            ensure_multiple_of = self.cfg.get('ensure_multiple_of', 14)
            width = self.cfg.get('width', None)
            height = self.cfg.get('height', None)
            resize_target = self.cfg.get('resize_target', False)
            Log.info(f"Using ensure_multiple_of {ensure_multiple_of}")
            Log.info(f"Using test width {width}")
            Log.info(f"Using test height {height}")
            Log.info(f"Resize target: {resize_target}")
            self.transform = Compose([
                Resize(
                    width=width,
                    height=height,
                    resize_target=resize_target,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=ensure_multiple_of,
                    resize_method='lower_bound',
                    image_interpolation_method=cv2.INTER_AREA,
                ),
                PrepareForNet()])
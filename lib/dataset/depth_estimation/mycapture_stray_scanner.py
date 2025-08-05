from lib.dataset.depth_estimation.depth_estimation import *
from os.path import join
import os
from torchvision.transforms import Compose
import pandas as pd
from tqdm import tqdm
import glob
from lib.dataset.depth_estimation.depth_estimation import Dataset as BaseDataset

class Dataset(BaseDataset):
    
    def build_metas(self):
        self.dataset_name = 'mycapture_stray_scanner'
        rgb_root = join(self.cfg.data_root, self.cfg.scene, 'rgb')
        self.rgb_files = sorted(glob.glob(join(rgb_root,'*.png')))
        self.low_files = [rgb_file.replace('/rgb/', '/depth/')  for rgb_file in self.rgb_files]
        self.conf_files = [rgb_file.replace('/rgb/', '/confidence/')  for rgb_file in self.rgb_files]
        frames = self.cfg.get('frames', [0, -1, 1])
        s, e, i = frames[0], frames[1], frames[2]
        e = min(e, len(self.rgb_files)) if e != -1 else len(self.rgb_files)
        self.rgb_files = self.rgb_files[s:e:i]
        self.low_files = self.low_files[s:e:i]
        self.conf_files = self.conf_files[s:e:i]
    
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
            
    def __getitem__(self, index):
        ret_dict = super().__getitem__(index)
        
        scene = self.rgb_files[index].split('/')[-3]
        direction = 'Left'
        
        # lowres_depth = np.asarray(imageio.imread(self.low_files[index]) / 1000.).astype(np.float32)
        lowres_depth = ret_dict['lowres_depth'][0]
        lowres_depth = Dataset.rotate_image(lowres_depth, direction)
        
        confidence = ret_dict['confidence'][0]
        confidence = Dataset.rotate_image(confidence, direction)
        
        image = ret_dict['image'].transpose(1, 2, 0)
        image = Dataset.rotate_image(image, direction)
        
        ret_dict.update({
            'image': image.transpose(2, 0, 1),
            'lowres_depth': lowres_depth[None],
            'confidence': confidence[None]
        })
        return ret_dict
    
    @staticmethod
    def rotate_image(img, direction):
        if direction == 'Up':
            pass
        elif direction == 'Left':
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif direction == 'Right':
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif direction == 'Down':
            img = cv2.rotate(img, cv2.ROTATE_180)
        else:
            raise Exception(f'No such direction (={direction}) rotation')
        return img
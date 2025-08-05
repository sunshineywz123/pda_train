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
import pandas as pd

class Dataset(BaseDataset):
    def build_metas(self):
        meta_file = self.cfg.data_root.replace('Validation', 'metadata.csv')
        meta_file = pd.read_csv(meta_file)
        self.scene_direction = {str(meta_file['video_id'][idx]): meta_file['sky_direction'][idx]
                           for idx in range(len(meta_file['video_id']))}
        scenes = sorted(os.listdir(self.cfg.data_root))
        self.rgb_files, self.dpt_files, self.low_files = [], [], []
        for scene in tqdm(scenes):
            rgb_files = sorted(glob.glob(os.path.join(self.cfg.data_root, scene, 'wide', '*.png')))
            dpt_files = sorted(glob.glob(os.path.join(self.cfg.data_root, scene, 'highres_depth', '*.png')))
            low_files = sorted(glob.glob(os.path.join(self.cfg.data_root, scene, 'lowres_depth', '*.png')))
            self.rgb_files += rgb_files
            self.dpt_files += dpt_files
            self.low_files += low_files
        frames = self.cfg.frames
        s,e,i = frames[0], frames[1], frames[2]
        e = min(e, len(self.rgb_files)) if e != -1 else len(self.rgb_files)
        self.rgb_files = self.rgb_files[s:e:i]
        self.dpt_files = self.dpt_files[s:e:i]
        self.low_files = self.low_files[s:e:i]
        self.metas = self.rgb_files
        Log.info(f'[Dataset]: {len(self.rgb_files)} frames in total.')
    
    def read_rgb(self, index):
        rgb_path = self.rgb_files[index]
        rgb = (np.asarray(imageio.imread(rgb_path)) / 255.).astype(np.float32)
        return rgb
    
    def read_dpt(self, index):
        dpt_path = self.dpt_files[index]
        dpt = np.asarray(imageio.imread(dpt_path)) / 1000.
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
        return dpt.astype(np.float32), (dpt != 0).astype(np.uint8)
    
    def read_rgb_name(self, index: int) -> str:
        # scene + camera + frame
        rgb_path = self.rgb_files[index]
        return rgb_path.split('/')[-3] + '_' +  os.path.basename(rgb_path)
    
    def __getitem__(self, index):
        ret_dict = super(Dataset, self).__getitem__(index)
        rgb = ret_dict['rgb'].transpose(1, 2, 0)
        dpt = ret_dict['dpt'][0]
        lowres_dpt = (imageio.imread(self.low_files[index]) / 1000.).astype(np.float32)
        conf = (imageio.imread(self.low_files[index].replace('lowres_depth', 'confidence'))).astype(np.uint8)

        tar_w = self.cfg.get('tar_w', 640)  # 640, 768, 1024, 1920
        assert (tar_w % 4 == 0)
        tar_h = int(tar_w / 4 * 3)
        if rgb.shape[0] != tar_h:
            rgb = cv2.resize(rgb, (tar_w, tar_h), interpolation=cv2.INTER_AREA).astype(np.float32)
            dpt = cv2.resize(dpt, (tar_w, tar_h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        
        scene = self.rgb_files[index].split('/')[-3]
        direction = self.scene_direction[scene]
        
        rgb = Dataset.rotate_image(rgb, direction)
        dpt = Dataset.rotate_image(dpt, direction)
        lowres_dpt = Dataset.rotate_image(lowres_dpt, direction)
        conf = Dataset.rotate_image(conf, direction)
        
        msk = dpt!=0.
        
        ret_dict.update({
            'rgb': rgb.transpose(2, 0, 1),
            'dpt': dpt[None],
            'lowres_dpt': lowres_dpt[None],
            'msk': msk[None].astype(np.uint8)
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
    
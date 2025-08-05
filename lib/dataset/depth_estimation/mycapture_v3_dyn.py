import glob

from lib.dataset.depth_estimation.depth_estimation import Dataset as BaseDataset
from lib.dataset.depth_estimation.depth_estimation import *
from os.path import join
import os
from torchvision.transforms import Compose
import json
from copy import deepcopy
from lib.utils.colmap_utils import cv_undistort_img, parse_dist, parse_ixt
from lib.utils.pylogger import Log
from tqdm import tqdm

from trdparties.colmap.read_write_model import read_cameras_binary, read_images_binary

class Dataset(BaseDataset):
    
    def build_metas(self):
        self.dataset_name = 'mycapture_v3_dyn'
        data_root = join(self.cfg.data_root, self.cfg.scene)
        colmap_path = self.cfg.get('colmap_path', 'colmap/sparse/0')
        if self.cfg.get('undistort', False):
            self.ixts, self.Ds, self.hws = [], [], []
        else:
            self.ixts = None
            self.Ds = None
            self.hws = None
        if os.path.exists(join(data_root, colmap_path)):
            Log.info(f"Using COLMAP sparse reconstruction for {self.cfg.scene}")
            images = read_images_binary(join(data_root, colmap_path, 'images.bin'))
            cameras = read_cameras_binary(join(data_root, colmap_path, 'cameras.bin'))
            rgb_paths = []
            depth_paths = []
            for im_id in images:
                img_name = images[im_id].name
                img_name_num = int(img_name.split('.')[0])
                if img_name_num < 30:
                    continue
                img_files = sorted(glob.glob(join(data_root, 'images_undist', '{:02d}/*.jpg'.format(img_name_num))))
                low_files = sorted(glob.glob(join(data_root, 'depth', '{:02d}/*.png'.format(img_name_num))))
                rgb_paths.append(img_files)
                depth_paths.append(low_files)
                if self.cfg.get('undistort', False):
                    camera_id = images[im_id].camera_id
                    camera = cameras[camera_id]
                    ixt, D = parse_ixt(camera), parse_dist(camera)
                    self.ixts.append(ixt)
                    self.Ds.append(D)
                    self.hws.append([camera.height, camera.width])
        else:
            raise NotImplementedError
        frame_len = len(rgb_paths[0])
        for i in range(len(rgb_paths)):
            assert(len(rgb_paths[i]) == frame_len, 'Frame length not consistent')
            assert(len(depth_paths[i]) == frame_len, 'Frame length not consistent')

        frame_sample = self.cfg.get('frames', [0, -1, 1])
        frame_sample[1] =  frame_len if frame_sample[1] == -1 else frame_sample[1]
        rgb_paths = [item[frame_sample[0]:frame_sample[1]:frame_sample[2]] for item in rgb_paths]
        lowres_paths = [item[frame_sample[0]:frame_sample[1]:frame_sample[2]] for item in depth_paths]

        ixts, Ds, hws = self.ixts, self.Ds, self.hws
        if self.ixts is not None:
            self.ixts = [ixts[i//len(rgb_paths[0])] for i in range(len(rgb_paths[0]) * len(rgb_paths))]
            self.Ds = [Ds[i//len(rgb_paths[0])] for i in range(len(rgb_paths[0]) * len(rgb_paths))]
            self.hws = [hws[i//len(rgb_paths[0])] for i in range(len(rgb_paths[0]) * len(rgb_paths))]

        self.rgb_files = []
        self.low_files = []
        for i in range(len(rgb_paths)):
            self.rgb_files += rgb_paths[i]
            self.low_files += lowres_paths[i]
        
    def read_rgb(self, index):
        img_path = self.rgb_files[index]
        start_time = time.time()
        rgb = cv2.imread(img_path)
        end_time = time.time()
        if self.ixts is not None and self.cfg.get('undistort', False):
            rgb = cv_undistort_img(rgb, self.ixts[index], self.Ds[index], self.hws[index])
        if end_time - start_time > 1: Log.warn(f'Long time to read {img_path}: {end_time - start_time}')
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return np.asarray(rgb / 255.).astype(np.float32)
        
    def read_rgb_name(self, index):
        return '__'.join(self.rgb_files[index].split('/')[-2:])
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
        self.dataset_name = 'mycapture_v3'
        data_root = join(self.cfg.data_root, self.cfg.scene)
        colmap_path = self.cfg.get('colmap_path', 'colmap/sparse/0')
        if os.path.exists(join(data_root, colmap_path)) and self.cfg.get('use_colmap', True):
            Log.info(f"Using COLMAP sparse reconstruction for {self.cfg.scene}")
            images = read_images_binary(join(data_root, colmap_path, 'images.bin'))
            images = {im_id:images[im_id] for im_id in images if 'for_' not in images[im_id].name}
            rgb_paths = [join(data_root, 'images', images[i].name) for i in images]
            lowres_paths = [join(data_root, 'depth', os.path.basename(images[i].name).replace('.jpg', '.png')) for i in images]
            if not os.path.exists(lowres_paths[-1]):
                rgb_paths = rgb_paths[:-1]
                lowres_paths = lowres_paths[:-1]
            if self.cfg.get('undistort', False):
                cameras = read_cameras_binary(join(data_root, colmap_path, 'cameras.bin'))
                camera_ids = [images[i].camera_id for i in images]
                cameras = [cameras[i] for i in camera_ids]
                self.ixts, self.Ds, self.hws = [], [], []
                for camera in cameras:
                    ixt, D = parse_ixt(camera), parse_dist(camera)
                    self.ixts.append(ixt)
                    self.Ds.append(D)
                    self.hws.append([camera.height, camera.width])
            else:
                self.ixts = None
                self.Ds = None
                self.hws = None
        else:
            Log.info(f"Using RGB-D images for {self.cfg.scene}")
            rgb_paths = sorted(glob.glob(join(data_root, 'rgb', '*.jpg')))
            rgb_paths += sorted(glob.glob(join(data_root, 'rgb', '*.png')))
            lowres_paths = sorted(glob.glob(join(data_root, 'depth', '*.png')))
            if len(lowres_paths) == len(rgb_paths) -1:
                rgb_paths = rgb_paths[:-1]
            elif len(lowres_paths) == len(rgb_paths) + 1:
                lowres_paths = lowres_paths[:-1]
            self.ixts = None
        frame_len = len(rgb_paths)
        frame_sample = self.cfg.get('frames', [0, -1, 1])
        frame_sample[1] =  frame_len if frame_sample[1] == -1 else frame_sample[1]
        rgb_paths = rgb_paths[frame_sample[0]:frame_sample[1]:frame_sample[2]]
        lowres_paths = lowres_paths[frame_sample[0]:frame_sample[1]:frame_sample[2]]

        self.rgb_files = rgb_paths
        self.low_files = lowres_paths
        if self.cfg.get('undistort_colmap', False):
            self.rgb_files = [path.replace('images', 'colmap/dense/images') for path in self.rgb_files]
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
        if self.cfg.get('scene', None) is not None:
            return self.rgb_files[index].split('/')[-1]
        else:
            return '__'.join(self.rgb_files[index].split('/')[-3:])
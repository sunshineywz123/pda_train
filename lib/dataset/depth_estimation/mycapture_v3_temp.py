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


im2cam = {
    'lidar_cam01/lidar_cam01.jpg': '30',
    'lidar_cam02/lidar_cam02.jpg': '31',
    'lidar_cam03/lidar_cam03.jpg': '32',
    'lidar_cam04/lidar_cam04.jpg': '33',
}

from trdparties.colmap.read_write_model import read_cameras_binary, read_images_binary

class Dataset(BaseDataset):
    
    def build_metas(self):
        self.dataset_name = 'mycapture_v3_temp'
        data_root = join(self.cfg.data_root, self.cfg.scene)
        colmap_path = self.cfg.get('colmap_path', 'colmap/sparse_rp/0_metric')
        if os.path.exists(join(data_root, colmap_path)):
            Log.info(f"Using COLMAP sparse reconstruction for {self.cfg.scene}")
            images = read_images_binary(join(data_root, colmap_path, 'images.bin'))
            cameras = read_cameras_binary(join(data_root, colmap_path, 'cameras.bin'))

            self.ixts, self.Ds, self.hws = [], [], []
            self.rgb_files, self.low_files = [], []
            for im_id in images:
                image = images[im_id]
                if 'lidar_cam' not in image.name:
                    continue
                camera = cameras[image.camera_id]
                ixt, D = parse_ixt(camera), parse_dist(camera)
                cam_id = im2cam[image.name]
                rgb_file = join(data_root, 'frame1', '{}_rgb.jpg'.format(cam_id))
                low_file = join(data_root, 'frame1', '{}_depth.png'.format(cam_id))
                self.low_files.append(low_file)
                self.rgb_files.append(rgb_file)
                self.ixts.append(ixt)
                self.Ds.append(D)
                self.hws.append([camera.height, camera.width])
        else:
            Log.info(f"Using RGB-D images for {self.cfg.scene}")
            rgb_paths = sorted(glob.glob(join(data_root, 'rgb', '*.jpg')))
            rgb_paths += sorted(glob.glob(join(data_root, 'rgb', '*.png')))
            lowres_paths = sorted(glob.glob(join(data_root, 'depth', '*.png')))
            if len(lowres_paths) == len(rgb_paths) -1:
                rgb_paths = rgb_paths[:-1]
            elif len(lowres_paths) == len(rgb_paths) + 1:
                lowres_paths = lowres_paths[:-1]
        
    def read_rgb(self, index):
        img_path = self.rgb_files[index]
        rgb = cv2.imread(img_path)
        rgb = cv_undistort_img(rgb, self.ixts[index], self.Ds[index], self.hws[index])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return np.asarray(rgb / 255.).astype(np.float32)
        
    def read_rgb_name(self, index):
        if self.cfg.get('scene', None) is not None:
            return self.rgb_files[index].split('/')[-1]
        else:
            return '__'.join(self.rgb_files[index].split('/')[-3:])
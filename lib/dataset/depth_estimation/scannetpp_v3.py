import ipdb
from lib.dataset.depth_estimation.depth_estimation import Dataset as BaseDataset
from lib.dataset.depth_estimation.depth_estimation import *
from os.path import join
import os
from torchvision.transforms import Compose
import json
from copy import deepcopy
from lib.utils.pylogger import Log
from tqdm import tqdm

from trdparties.colmap.read_write_model import read_cameras_binary, read_images_binary

class Dataset(BaseDataset):
    
    def build_metas(self):
        self.dataset_name = 'scannetpp'
        
        metadata = json.load(open(self.cfg.split_path))
        rgb_paths = metadata['rgb_files']
        dpt_paths = metadata['depth_files']
        lowres_paths = metadata['lowres_files']
        if 'mesh_depth_files' in metadata:
            mesh_depth_paths = metadata['mesh_depth_files']
            # sem_paths = metadata['sem_files']
        

        self.ixt = None
        self.dist = None
        if self.cfg.get('scene', None) is not None:
            scene = self.cfg.get('scene')
            rgb_paths = [path for path in rgb_paths if self.cfg.scene in path]
            dpt_paths = [path for path in dpt_paths if self.cfg.scene in path]
            lowres_paths = [path for path in lowres_paths if self.cfg.scene in path]
            if 'mesh_depth_files' in metadata:
                mesh_depth_paths = [path for path in mesh_depth_paths if self.cfg.scene in path]
                # sem_paths = [path for path in sem_paths if self.cfg.scene in path]
            
            if self.cfg.get('undistort', False):
                colmap_path = f'/mnt/bn/haotongdata/Datasets/scannetpp/data/{scene}/merge_dslr_iphone/colmap/sparse_render_rgb'
                cameras = read_cameras_binary(join(colmap_path, 'cameras.bin'))
                ixt = np.eye(3) 
                ixt[0, 0], ixt[1, 1], ixt[0, 2], ixt[1, 2] = cameras[2].params[:-4]
                dist =  np.zeros(5)
                dist[:4] = cameras[2].params[-4:]
                self.ixt = ixt
                self.dist = dist
        
        frame_len = len(rgb_paths)
        frame_sample = self.cfg.get('frames', [0, -1, 1])
        frame_sample[1] =  frame_len if frame_sample[1] == -1 else frame_sample[1]
        
        rgb_paths = rgb_paths[frame_sample[0]:frame_sample[1]:frame_sample[2]]
        dpt_paths = dpt_paths[frame_sample[0]:frame_sample[1]:frame_sample[2]]
        lowres_paths = lowres_paths[frame_sample[0]:frame_sample[1]:frame_sample[2]]
        if 'mesh_depth_files' in metadata:
            mesh_depth_paths = mesh_depth_paths[frame_sample[0]:frame_sample[1]:frame_sample[2]]
            # sem_paths = sem_paths[frame_sample[0]:frame_sample[1]:frame_sample[2]]

        self.rgb_files = rgb_paths
        self.depth_files = dpt_paths
        self.low_files = lowres_paths

        if self.cfg.get('undistort_colmap', False):
            self.rgb_files = [path.replace('images/iphone', 'colmap/dense/images/iphone') for path in self.rgb_files]
        if self.cfg.get('undistort', False):
            scene_info = {}
            for rgb_file in self.rgb_files:
                scene = rgb_file.split('/')[-5]
                if scene not in scene_info:
                    colmap_path = f'/mnt/bn/haotongdata/Datasets/scannetpp/data/{scene}/merge_dslr_iphone/colmap/sparse_render_rgb'
                    cameras = read_cameras_binary(join(colmap_path, 'cameras.bin'))
                    assert(cameras[2].params[2] == 960., f'Wrong camera center: {cameras[2].params[2]}')
                    ixt = np.eye(3)
                    ixt[0, 0], ixt[1, 1], ixt[0, 2], ixt[1, 2] = cameras[2].params[:-4]
                    dist =  np.zeros(5)
                    dist[:4] = cameras[2].params[-4:]
                    scene_info[scene] = (ixt, dist)
            self.scene_info = scene_info
        else:
            self.scene_info = None
        if 'mesh_depth_files' in metadata:
            self.mesh_depth_files = mesh_depth_paths
        self.split = self.cfg.get('split', 'train')
            # self.sem_files = sem_paths
    # def build_transforms(self):
    #     transforms = []
    #     width = self.cfg.get('width', None)
    #     height = self.cfg.get('height', None)
    #     resize_target = self.cfg.get('resize_target', False)
    #     ensure_multiple_of = self.cfg.get('ensure_multiple_of', 8)
    #     Log.info(f"Using {self.cfg.split} width: {width}")
    #     Log.info(f"Using {self.cfg.split} height: {height}")
    #     Log.info(f"Resize {self.cfg.split} target: {resize_target}")
    #     Log.info(f"Using {self.cfg.split} ensure_multiple_of: {ensure_multiple_of}")
    #     resize_layer = Resize(
    #         width=width,
    #         height=height,
    #         resize_target=resize_target,
    #         keep_aspect_ratio=True,
    #         ensure_multiple_of=ensure_multiple_of,
    #         resize_method='lower_bound',
    #         image_interpolation_method=cv2.INTER_AREA
    #     )
    #     transforms.append(resize_layer)
    #     transforms.append(PrepareForNet())
    #     self.transform = Compose(transforms)
    
    def read_depth(self, index, depth=None):
        if self.split == 'train':
            depth, valid_mask = super().read_depth(index, depth)
        else:
            depth = np.asarray(imageio.imread(self.mesh_depth_files[index]) / 1000.).astype(np.float32)
            valid_mask = np.logical_and(depth > 0.05, depth < 5)
            valid_mask = valid_mask.astype(np.uint8)
        return depth, valid_mask

    def read_rgb(self, index):
        img_path = self.rgb_files[index]
        start_time = time.time()
        if not os.path.exists(img_path):
            Log.error(f'File not found: {img_path}')
            raise FileNotFoundError
        rgb = cv2.imread(img_path)
        end_time = time.time()
        if self.ixt is not None:
            new_ixt, roi = cv2.getOptimalNewCameraMatrix(self.ixt, self.dist, (rgb.shape[1], rgb.shape[0]), 1)
            h_orig, w_orig = rgb.shape[:2]
            rgb = cv2.undistort(rgb, self.ixt, self.dist, newCameraMatrix=new_ixt)
            x, y, w, h = roi
            rgb = rgb[y : y + h, x : x + w]
            rgb = cv2.resize(rgb, (w_orig, h_orig), interpolation=cv2.INTER_AREA)
        elif self.scene_info is not None:
            scene = img_path.split('/')[-5]
            ixt, dist = self.scene_info[scene]
            ixt = ixt.copy()
            ixt[:2, 2] -= 0.5
            new_ixt, roi = cv2.getOptimalNewCameraMatrix(ixt, dist, (rgb.shape[1], rgb.shape[0]), 1)
            h_orig, w_orig = rgb.shape[:2]
            rgb = cv2.undistort(rgb, ixt, dist, newCameraMatrix=new_ixt)
            x, y, w, h = roi
            rgb = rgb[y : y + h, x : x + w]
            rgb = cv2.resize(rgb, (w_orig, h_orig), interpolation=cv2.INTER_AREA)
        if end_time - start_time > 1: Log.warn(f'Long time to read {img_path}: {end_time - start_time}')
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return np.asarray(rgb / 255.).astype(np.float32)
        
    def check_shape(self, rgb, dpt):
        pass

    def read_rgb_name(self, index):
        if self.cfg.get('scene', None) is not None:
            return self.rgb_files[index].split('/')[-1]
        elif self.cfg.get('undistort_colmap', False):
            return '__'.join(self.rgb_files[index].replace('colmap/dense/images/iphone', 'images/iphone').split('/')[-5:])
        else:
            return '__'.join(self.rgb_files[index].split('/')[-5:])
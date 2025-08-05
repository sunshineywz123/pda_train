from lib.dataset.depth_estimation.depth_estimation import Dataset as BaseDataset
from lib.dataset.depth_estimation.depth_estimation import *
from os.path import join
import os
from torchvision.transforms import Compose
import json
from copy import deepcopy
from lib.utils.pylogger import Log
from tqdm import tqdm
import pickle
class Dataset(BaseDataset):
    
    def build_metas(self):

        self.dataset_name = 'shift'
        self.split = self.cfg.split
        if self.split=='train':
            with open('/home/nas/users/pangbo/pl_htcode/senseguide/lightwheel_occ_infos_train.pkl','rb') as f: data_infos= pickle.load(f)
        else:
            with open('/home/nas/users/pangbo/pl_htcode/senseguide/lightwheel_occ_infos_val.pkl','rb') as f: data_infos= pickle.load(f)

        rgb_paths=[]
        dpt_paths=[]
        for sample_data in data_infos['infos']:
            cam = sample_data['cams']['CAM_FRONT']
            rgb_path=os.path.join('/home/nas/users/pangbo/pl_htcode/senseguide', cam['cam_path'])
            depth_path=os.path.join('/home/nas/users/pangbo/pl_htcode/senseguide', cam['depth_path'])
            if os.path.exists(rgb_path) and os.path.exists(depth_path):
                rgb_paths.append(rgb_path)
                dpt_paths.append(depth_path)
                occ_path = os.path.join('/home/nas/users/pangbo/pl_htcode/senseguide', sample_data['occ_path'])
                import pdb;pdb.set_trace()

        # sample_data=data_infos['infos'][0]
        # cam = sample_data['cams']['CAM_FRONT']
        # rgb_path=os.path.join('/home/nas/users/pangbo/pl_htcode/senseguide', cam['cam_path'])
        # rgb=cv2.imread(rgb_path)
        # depth_path=os.path.join('/home/nas/users/pangbo/pl_htcode/senseguide', cam['depth_path'])
        # depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        # depth = depth_img[:,:,0] + (depth_img[:,:,1] * 256)

        # splits = json.load(open(self.cfg.split_path))
        # data_root = self.cfg.data_root
        # rgb_paths = splits['rgb_files']
        # dpt_paths = splits['depth_files']
        # low_dpt_paths = splits['lowres_files']
        
        
        # rgb_paths = [join(data_root, path) for path in tqdm(rgb_paths)]
        # dpt_paths = [join(data_root, path) for path in tqdm(depth_paths)]
        # low_dpt_paths = [join(data_root, path) for path in tqdm(low_depth_paths)]

        frame_len = len(rgb_paths) 
        frame_sample = self.cfg.get('frames', [0, -1, 1])
        frame_sample[1] =  frame_len if frame_sample[1] == -1 else frame_sample[1]
        
        rgb_paths = rgb_paths[frame_sample[0]:frame_sample[1]:frame_sample[2]]
        dpt_paths = dpt_paths[frame_sample[0]:frame_sample[1]:frame_sample[2]]
        # low_dpt_paths = low_dpt_paths[frame_sample[0]:frame_sample[1]:frame_sample[2]]
        
        self.rgb_files = rgb_paths
        self.depth_files = dpt_paths
        # self.low_files = low_dpt_paths
        # self.__DEPTH_C = np.array(1000.0 / (256 * 256 * 256 - 1), np.float32)
    def read_rgb(self, index):
        pass

    def read_depth(self, index, depth=None):

        depth_img = cv2.imread(self.depth_files[index], cv2.IMREAD_UNCHANGED)
        depth = depth_img[:,:,0] + (depth_img[:,:,1] * 256)
        depth = depth * 0.01

        valid_mask = (depth < 80.)
        depth[~valid_mask] = 80.

        return depth, np.ones_like(depth).astype(np.uint8)

    
    def read_low_depth(self, file, index=None):
        return None
        # depth = cv2.imread(file)
        # depth = (depth[:, :, 0] * 256. * 256. + depth[:, :, 1] * 256. + depth[:, :, 2]) * self.__DEPTH_C
        # ill_mask = (depth > 80.)
        # depth[ill_mask] = 80.
        # return depth
        # import ipdb; ipdb.set_trace()
        # depth = cv2.imread(file)
        # depth = (depth[:, :, 0] * 256. * 256. + depth[:, :, 1] * 256. + depth[:, :, 2]) * self.__DEPTH_C
        
        # lidar_mask = (depth != 0.) & (depth < 80.)
        # gt_depth, gt_mask = self.read_depth(index)
        # # new_mask = gt_mask & lidar_mask
        # output_depth = np.zeros_like(gt_depth)
        # output_depth[lidar_mask] = gt_depth[lidar_mask]
        # depth = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        # depth = (depth[:, :, 0] * 256. * 256. + depth[:, :, 1] * 256. + depth[:, :, 2]) * self.__DEPTH_C
        # ixt = np.array([[640., 640.]])
        
        
        # return output_depth
    

# def generate_lidar_depth(depth, ixt, tar_lines = 64, reserve_ratio = 0.5):
    
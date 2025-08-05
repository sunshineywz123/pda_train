import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
import pandas as pd
import h5py
import cv2
import json

from lib.utils.parallel_utils import parallel_execution


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    return args

# Define some global variables for hypersim
hypersim_cam_dir = '_detail/cam_{subscene}'
hypersim_img_dir = 'images/scene_cam_{subscene}_final_preview'
hypersim_dpt_dir = 'images/scene_cam_{subscene}_geometry_hdf5'
hypersim_img_pattern = 'frame.{frame:04d}.tonemap.jpg'
hypersim_dpt_pattern = 'frame.{frame:04d}.depth_meters.hdf5'


hypersim_meta_dir = 'data/pl_htcode/processed_datasets/HyperSim'
hypersim_meta_camera = 'metadata_camera_parameters.csv'
hypersim_dir = 'data/pl_htcode/datasets/HyperSim/all'

# meta_camera = pd.read_csv(join(hypersim_meta_dir, hypersim_meta_camera), index_col="scene_name")

    
def main(args):
    valid_rgb_paths = []
    valid_dpt_paths = []
    valid_train_rgb_paths = []
    valid_train_dpt_paths = []
    valid_test_rgb_paths = []
    valid_test_dpt_paths = []
    
    rgb_paths = []
    dpt_paths = []
    train_rgb_paths = []
    train_dpt_paths = []
    test_rgb_paths = []
    test_dpt_paths = []
    
    
    splits = pd.read_csv(join(hypersim_meta_dir, 'metadata_images_split_scene_v1.csv'))
    for idx in tqdm(range(len(splits['scene_name']))):
        data = splits.iloc[idx]
        scene_name, camera_name, frame_id, include, split_partition_name = data['scene_name'], data['camera_name'], data['frame_id'], data['included_in_public_release'], data['split_partition_name']
        if not include: continue
        
        rgb_path = join(f'all/{scene_name}/images/scene_{camera_name}_final_preview', f'frame.{frame_id:04d}.tonemap.jpg')
        dpt_path = join(f'all/{scene_name}/images/scene_{camera_name}_geometry_hdf5', f'frame.{frame_id:04d}.depth.npz')
        
        rgb_paths.append(rgb_path); dpt_paths.append(dpt_path)
        if split_partition_name == 'train':
            train_rgb_paths.append(rgb_path)
            train_dpt_paths.append(dpt_path)
        else:
            test_rgb_paths.append(rgb_path)
            test_dpt_paths.append(dpt_path)
        
        depth = np.load(join('data/pl_htcode/processed_datasets/HyperSim', dpt_path))['data']
        if np.isinf(depth).mean() > 0.05 or np.isnan(depth).mean() > 0.05: 
            continue
        msk = (~np.isinf(depth)) & (~np.isnan(depth))
        if (depth[msk] < 0.05).mean() > 0.05: # 如果有超过0.05的区域depth都小于5cm, 则不要
            continue
        valid_rgb_paths.append(rgb_path); valid_dpt_paths.append(dpt_path)
        if split_partition_name == 'train': 
            valid_train_rgb_paths.append(rgb_path)
            valid_train_dpt_paths.append(dpt_path)
        else:
            valid_test_rgb_paths.append(rgb_path)
            valid_test_dpt_paths.append(dpt_path)
    # output_file_path = join('data/pl_htcode/processed_datasets/HyperSim/metadata_splits_ht.json')
    # json.dump({'rgb_paths': rgb_paths, 
    #            'dpt_paths': dpt_paths, 
    #            'train_rgb_paths': train_rgb_paths, 
    #            'train_dpt_paths': train_dpt_paths, 
    #            'test_rgb_paths': test_rgb_paths, 
    #            'test_dpt_paths': test_dpt_paths}, 
    #           open(output_file_path, 'w'))
    output_file_path = join('data/pl_htcode/processed_datasets/HyperSim/metadata_valid_0.05_splits_ht_new.json')
    print(f'Valid RGB paths: {len(valid_rgb_paths)}')
    print(f'Valid DPT paths: {len(valid_dpt_paths)}')
    print(f'Valid Train RGB paths: {len(valid_train_rgb_paths)}')
    print(f'Valid Train DPT paths: {len(valid_train_dpt_paths)}')
    print(f'Valid Test RGB paths: {len(valid_test_rgb_paths)}')
    print(f'Valid Test DPT paths: {len(valid_test_dpt_paths)}')
    json.dump({'rgb_paths': valid_rgb_paths, 
               'dpt_paths': valid_dpt_paths, 
               'train_rgb_paths': valid_train_rgb_paths, 
               'train_dpt_paths': valid_train_dpt_paths, 
               'test_rgb_paths': valid_test_rgb_paths, 
               'test_dpt_paths': valid_test_dpt_paths}, 
              open(output_file_path, 'w'))
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
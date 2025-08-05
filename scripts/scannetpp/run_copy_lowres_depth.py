import numpy as np
import os
import imageio
import json
from tqdm import tqdm
import cv2
import sys
sys.path.append('.')
from lib.utils.parallel_utils import parallel_execution

def resize_copy(src_path, tar_path):
    depth = np.asarray(imageio.imread(src_path) / 1000.)
    depth = cv2.resize(depth, (1920, 1440), interpolation=cv2.INTER_LINEAR)
    np.savez_compressed(tar_path, data=np.round(depth, 3))

scenes = ['7b6477cb95', 'c50d2d1d42', 'cc5237fd77', 'acd95847c5', '31a2c91c43', 'e7af285f7d', '286b55a2bf', '7bc286c1b6']
input_dir = '/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/datasets/scannetpp/data'
output_dir = '/mnt/bn/haotongdata/home/linhaotong/workspaces/scannetpp_benchmark'
exp = 'lowres_lidar_bilinear_upsample'

for scene in tqdm(scenes):
    depth_dir = os.path.join(input_dir, scene, 'iphone/depth')
    depth_files = sorted(os.listdir(depth_dir))[::10]
    src_paths, tar_paths = [], []
    for depth_file in tqdm(depth_files):
        src_path = os.path.join(depth_dir, depth_file)
        tar_path = os.path.join(output_dir, f'{scene}_{exp}/depth', depth_file[6:12] + '.npz')
        os.makedirs(os.path.dirname(tar_path), exist_ok=True)
        src_paths.append(src_path)
        tar_paths.append(tar_path)
        # depth = np.asarray(imageio.imread(src_path) / 1000.)
        # depth = cv2.resize(depth, (1920, 1440), interpolation=cv2.INTER_LINEAR)
        # np.savez_compressed(tar_path, data=np.round(depth, 3))
    parallel_execution(
        src_paths,
        tar_paths,
        action=resize_copy,
        print_progress=True
    )
    config = {
        'scene_name': scene,
        'is_disparity': False,
        'is_metric': True,
        'align_methods': ['gt'],
        'method': exp
    }
    json_output_path = os.path.join(output_dir, f'{scene}_{exp}/config.json')
    with open(json_output_path, 'w') as f:
        json.dump(config, f)

import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')

from lib.utils.parallel_utils import parallel_execution


def process_one_scene(scene_name, output_dir):
    
    length = len(os.listdir(join(output_dir, scene_name, 'sensor_data'))) // 3
    
    for i in range(length):
        src_path = join(output_dir, scene_name, 'sensor_data', 'frame-{:05d}.color.jpg'.format(i))
        tar_path = join(output_dir, scene_name, 'sensor_data', 'frame-{:06d}.color.jpg'.format(i))
        if os.path.exists(src_path):
            os.system('mv {} {}'.format(src_path, tar_path))
        src_path = join(output_dir, scene_name, 'sensor_data', 'frame-{:05d}.depth.png'.format(i))
        tar_path = join(output_dir, scene_name, 'sensor_data', 'frame-{:06d}.depth.png'.format(i))
        if os.path.exists(src_path):
            os.system('mv {} {}'.format(src_path, tar_path))
        src_path = join(output_dir, scene_name, 'sensor_data', 'frame-{:05d}.pose.txt'.format(i))
        tar_path = join(output_dir, scene_name, 'sensor_data', 'frame-{:06d}.pose.txt'.format(i))
        if os.path.exists(src_path):
            os.system('mv {} {}'.format(src_path, tar_path))
    # sensors_dir/scene_name_00/color/0.jpg -> output_dir/scene_name/sensor_data/frame-00000.color.jpg
    # sensors_dir/scene_name_00/depth/0.png -> output_dir/scene_name/sensor_data/frame-00000.depth.png
    # sensors_dir/scene_name_00/pose/0.txt -> output_dir/scene_name/sensor_data/frame-00000.pose.txt
    # note that original pose include 4x4, but the final should be 3x4, simply discard the last row

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    return args

def main(args):
    # sensors_dir = '/nas/datasets/scannet/scannet'
    # meta_dir = '/nas/datasets/scannet/scannet/scans_test'
    # output_dir = '/mnt/data/Datasets/scannet_simplerecon/scans_test'
    # scenes = [scene.strip() for scene in open('/nas/datasets/scannet/splits/scannetv2_test.txt').readlines()]
    # for scene in tqdm(scenes[1:], desc='processing test scenes'):
    #     process_one_scene(scene[:-3], sensors_dir, meta_dir, output_dir)
        
    # sensors_dir = '/nas/datasets/scannet/output'
    # meta_dir = '/nas/datasets/scannet/scannet/scans'
    # output_dir = '/mnt/data/Datasets/scannet_simplerecon/scans'
    # scenes = [scene.strip() for scene in open('/nas/datasets/scannet/splits/scannetv2_val.txt').readlines()]
    # for scene in tqdm(scenes[1:], desc='processing val scenes'):
    #     process_one_scene(scene[:-3], sensors_dir, meta_dir, output_dir)
    output_dir = '/mnt/data/Datasets/scannet_simplerecon/scans_test'
    scenes = [scene.strip()[:-3] for scene in open('/nas/datasets/scannet/splits/scannetv2_test.txt').readlines()]
    parallel_execution(scenes, 
                       output_dir,
                       action=process_one_scene, num_processes=8, print_progress=True)
    output_dir = '/mnt/data/Datasets/scannet_simplerecon/scans'
    scenes = [scene.strip()[:-3] for scene in open('/nas/datasets/scannet/splits/scannetv2_train.txt').readlines()]
    parallel_execution(scenes[1:], 
                       output_dir,
                       action=process_one_scene, num_processes=8, print_progress=True)
    output_dir = '/mnt/data/Datasets/scannet_simplerecon/scans'
    scenes = [scene.strip()[:-3] for scene in open('/nas/datasets/scannet/splits/scannetv2_val.txt').readlines()]
    parallel_execution(scenes, 
                       output_dir,
                       action=process_one_scene, num_processes=8, print_progress=True)

    


        
if __name__ == '__main__':
    args = parse_args()
    main(args)
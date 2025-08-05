import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')

from lib.utils.parallel_utils import parallel_execution


def process_one_scene(scene_name, output_dir):
    src_path = join(output_dir, scene_name)
    tar_path = join(output_dir, scene_name + '_00')
    if os.path.exists(src_path):
        os.system('mv {} {}'.format(src_path, tar_path))
    
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
    for scene in tqdm(scenes, desc='processing test scenes'):
        process_one_scene(scene, output_dir)
    # parallel_execution(scenes, 
    #                    output_dir,
    #                    action=process_one_scene, num_processes=8, print_progress=True)
    output_dir = '/mnt/data/Datasets/scannet_simplerecon/scans'
    scenes = [scene.strip()[:-3] for scene in open('/nas/datasets/scannet/splits/scannetv2_train.txt').readlines()]
    for scene in tqdm(scenes, desc='processing train scenes'):
        process_one_scene(scene, output_dir)
    # parallel_execution(scenes[1:], 
    #                    output_dir,
    #                    action=process_one_scene, num_processes=8, print_progress=True)
    output_dir = '/mnt/data/Datasets/scannet_simplerecon/scans'
    scenes = [scene.strip()[:-3] for scene in open('/nas/datasets/scannet/splits/scannetv2_val.txt').readlines()]
    for scene in tqdm(scenes, desc='processing val scenes'):
        process_one_scene(scene, output_dir)
    # parallel_execution(scenes, 
    #                    output_dir,
    #                    action=process_one_scene, num_processes=8, print_progress=True)

    


        
if __name__ == '__main__':
    args = parse_args()
    main(args)
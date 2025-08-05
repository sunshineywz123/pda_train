import json
import os
from os.path import join
import argparse
import sys
import ipdb
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from lib.utils.pylogger import Log
from lib.utils.parallel_utils import parallel_execution
import time

from trdparties.colmap.read_write_model import read_images_binary
sys.path.append('.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scannetpp_dir', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp')
    parser.add_argument('--input', type=str, default='/mnt/bn/haotongdata/home/linhaotong/projects/zipnerf-pytorch/exp/scannetpp_all_0610')
    parser.add_argument('--output_dir', type=str, default='/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/txts/scannetpp')
    args = parser.parse_args()
    return args

def get_processing_status(scene):
    if os.path.exists(join(args.input, scene, 'test_preds')): return 'processed'
    if os.path.exists(join(args.input, scene, 'checkpoints')): return 'processing'
    if os.path.exists(join(args.input, scene)): return 'error'
    return 'other'

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    train_scenes = open(join(args.scannetpp_dir, 'splits/nvs_sem_train.txt')).readlines()
    val_scenes = open(join(args.scannetpp_dir, 'splits/nvs_sem_val.txt')).readlines()
    scenes = train_scenes + val_scenes
    scenes = [scene.strip() for scene in scenes]
    scenes = sorted(scenes)
    scene_infos = {}
    processed_scene, processing_scenes, error_scenes, other_scenes = [], [], [], []
    for scene in tqdm(scenes):
        process_status = get_processing_status(scene)
        if process_status == 'processed': processed_scene.append(scene)
        elif process_status == 'processing': processing_scenes.append(scene)
        elif process_status == 'other': other_scenes.append(scene)
        else: 
            os.system(f'sudo rm -rf {join(args.input, scene)}')
            error_scenes.append(scene)
        
    with open(join(args.output_dir, 'processed_scene.txt'), 'w') as f:
        for scene in processed_scene:
            f.write(scene + '\n')
    Log.info(f'Saved {args.output_dir}/processed_scene.txt')
    
    with open(join(args.output_dir, 'processing_scene.txt'), 'w') as f:
        for scene in processing_scenes:
            f.write(scene + '\n')
    
    with open(join(args.output_dir, 'error_scene.txt'), 'w') as f:
        for scene in error_scenes:
            f.write(scene + '\n')
            
    with open(join(args.output_dir, 'other_scene.txt'), 'w') as f:
        for scene in other_scenes:
            f.write(scene + '\n')
        
    
    
    

    

if __name__ == '__main__':
    args = parse_args()
    main(args)
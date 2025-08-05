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
    parser.add_argument('--input', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp')
    parser.add_argument('--output_dir', type=str, default='/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/txts/scannetpp')
    args = parser.parse_args()
    return args

    
def get_num_images(scene):
    scannetpp_dir = f'/mnt/bn/haotongdata/Datasets/scannetpp/data/{scene}/merge_dslr_iphone/images'
    num_iphone, num_dslr = len(os.listdir(join(scannetpp_dir, 'iphone'))), len(os.listdir(join(scannetpp_dir, 'dslr')))
    return num_iphone, num_dslr

def get_regi_images(scene):
    scannetpp_dir = f'/mnt/bn/haotongdata/Datasets/scannetpp/data/{scene}/merge_dslr_iphone/colmap/sparse/0'
    if not os.path.exists(scannetpp_dir): return 0, 0
    images = read_images_binary(join(scannetpp_dir, 'images.bin'))
    num_iphone, num_dslr = 0, 0
    for k in images:
        im = images[k]
        if 'iphone' in im.name: num_iphone += 1
        elif 'dslr' in im.name: num_dslr += 1
        else: print(f"Unknown camera: {im.name}")
    return num_iphone, num_dslr

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    train_scenes = open(join(args.input, 'splits/nvs_sem_train.txt')).readlines()
    val_scenes = open(join(args.input, 'splits/nvs_sem_val.txt')).readlines()
    scenes = train_scenes + val_scenes
    scenes = [scene.strip() for scene in scenes]
    scenes = sorted(scenes)
    
    scene_infos = {}
    for scene in tqdm(scenes):
        num_iphone, num_dslr = get_num_images(scene)
        num_regi_iphone, num_regi_dslr = get_regi_images(scene)
        
        scene_infos[scene] = {
            'num_iphone': num_iphone,
            'num_dslr': num_dslr,
            'num_regi_iphone': num_regi_iphone,
            'num_regi_dslr': num_regi_dslr,
            'regi_ratio': (num_regi_iphone + num_regi_dslr) / (num_iphone + num_dslr)
        }
        
    # 按照regi_ratio排序
    sorted_scene_infos = sorted(scene_infos.items(), key=lambda x: x[1]['regi_ratio'], reverse=True)
    with open(join(args.output_dir, 'regi_ratio_scene_infos.json'), 'w') as f:
        for scene_info in sorted_scene_infos:
            f.write(json.dumps(scene_info) + '\n')
    Log.info(f'Saved {args.output_dir}/regi_ratio_scene_infos.json')
    
    sorted_scene_infos = sorted(scene_infos.items(), key=lambda x: x[1]['num_regi_iphone'], reverse=True)
    with open(join(args.output_dir, 'regi_iphone_scene_infos.json'), 'w') as f:
        for scene_info in sorted_scene_infos:
            f.write(json.dumps(scene_info) + '\n')
    Log.info(f'Saved {args.output_dir}/regi_iphone_scene_infos.json')
            
    sorted_scene_infos = sorted(scene_infos.items(), key=lambda x: x[1]['num_regi_dslr'], reverse=True)
    with open(join(args.output_dir, 'regi_dslr_scene_infos.json'), 'w') as f:
        for scene_info in sorted_scene_infos:
            f.write(json.dumps(scene_info) + '\n')
    Log.info(f'Saved {args.output_dir}/regi_dslr_scene_infos.json')
        
    
    
    

    

if __name__ == '__main__':
    args = parse_args()
    main(args)
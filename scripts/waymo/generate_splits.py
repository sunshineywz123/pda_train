import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
import glob

def main(args):
    root_dir = args.input
    seq = args.seq
    images = [f for f in sorted(os.listdir(join(root_dir, seq, 'images'))) if '_0.' in f]
    split = {'rgb_files': [],
             'depth_files': [],
             'lowres_files': []}
    
    split['rgb_files'].extend([join(root_dir, seq, 'images', f) for f in images])
    split['depth_files'].extend([join(root_dir, seq, 'lidar_depth', f.replace('.png', '.npy')) for f in images])
    split['lowres_files'].extend([join(root_dir, seq, 'lidar_depth', f.replace('.png', '.npy')) for f in images])
    os.makedirs(args.output, exist_ok=True)
    import json 
    with open(join(args.output, seq + '.json'), 'w') as f:
        json.dump(split, f)
            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/bn/haotongdata3/Datasets/waymo')
    parser.add_argument('--output', type=str, default='data/pl_htcode/processed_datasets/waymo')
    parser.add_argument('--seq', type=str, default='019')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)
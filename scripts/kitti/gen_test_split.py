import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm

from lib.utils.pylogger import Log
sys.path.append('.')

def main(args):
    # scenes = sorted(os.listdir(args.input + '/raw'))
    split = {'rgb_files': [],
             'depth_files': [],
             'lowres_files': []
             }
    tag = args.tag
    images = sorted(os.listdir(join(args.input, 'test_depth_completion_anonymous', 'image')))
    depth = sorted(os.listdir(join(args.input, 'test_depth_completion_anonymous', 'velodyne_raw')))
    lowdepth = sorted(os.listdir(join(args.input, 'test_depth_completion_anonymous', 'velodyne_raw')))
    split['rgb_files'].extend([join('test_depth_completion_anonymous', 'image', f) for f in images])
    split['depth_files'].extend([join('test_depth_completion_anonymous', 'velodyne_raw', f) for f in depth])
    split['lowres_files'].extend([join('test_depth_completion_anonymous', 'velodyne_raw', f) for f in lowdepth])
    os.makedirs(args.output, exist_ok=True)
    import json 
    output = join(args.output, f'{tag}_split.json')
    with open(output, 'w') as f:
        json.dump(split, f)
    Log.info(f'Saved {output}')
            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/bn/haotongdata/Datasets/kitti_completion')
    parser.add_argument('--output', type=str, default='data/pl_htcode/processed_datasets/kitti')
    parser.add_argument('--tag', type=str, default='test')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
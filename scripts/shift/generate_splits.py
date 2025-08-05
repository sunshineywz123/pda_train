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
    tag = args.tag
    scenes = sorted(os.listdir(join(root_dir, tag, 'front')))
    split = {'rgb_files': [],
             'depth_files': [],
             'lowres_files': []}
    
    
    for scene in tqdm(scenes):
        scene_dir = join(root_dir, tag, 'front', scene)
        imgs = sorted(glob.glob(join(scene_dir, '*_img_front.jpg')))
        depths = sorted(glob.glob(join(scene_dir, '*_depth_front.png')))
        lowdepths = sorted(glob.glob(join(scene_dir, '*_lidar_front.png')))
        assert len(imgs) == len(depths)
        assert len(imgs) == len(lowdepths)
        split['rgb_files'].extend(imgs)
        split['depth_files'].extend(depths)
        split['lowres_files'].extend(lowdepths)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    import json 
    with open(args.output, 'w') as f:
        json.dump(split, f)
            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/bn/haotongdata/Datasets/shift/discrete/images')
    parser.add_argument('--output', type=str, default='data/pl_htcode/processed_datasets/shift/val_split.json')
    parser.add_argument('--tag', type=str, default='val')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)
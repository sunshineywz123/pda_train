import argparse
import os
import sys
from os.path import join

import numpy as np
from tqdm import tqdm

sys.path.append('.')
import glob


def main(args):
    root_dir = args.input
    depth = args.input_depth
    scenes = sorted(os.listdir(join(root_dir)))
    depths = sorted(os.listdir(join(depth)))
    print(len(scenes), len(depths))
    # 取交集
    scene_set = set(scenes)
    depth_set = set(depths)
    common = sorted(list(scene_set & depth_set))
    if len(scenes) != len(depths):
        print(f"scenes和depths长度不等，取交集，交集数量: {len(common)}")
    scenes = common
    assert len(scenes) == len(depths)
    split = {'rgb_files': [],
             'depth_files': []}
    
    
    for scene in tqdm(scenes):
        scene_dir = join(root_dir,scene)
        depth_dir = join(depth,scene)
        imgs = sorted(glob.glob(join(scene_dir, '*_img_front.jpg')))
        depths_imgs = sorted(glob.glob(join(depth_dir, '*_depth_front.png')))
        # lowdepths = sorted(glob.glob(join(scene_dir, '*_lidar_front.png')))
        print(scene,len(imgs),len(depths_imgs))
        if not len(imgs) == len(depths_imgs):
            print(f"Warning: {scene} has different number of images and depth files: {len(imgs)} vs {len(depths_imgs)}")
            continue
        # assert len(imgs) == len(lowdepths)
        split['rgb_files'].extend(imgs)
        split['depth_files'].extend(depths_imgs)
        # split['lowres_files'].extend(lowdepths)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    import json 
    with open(args.output, 'w') as f:
        json.dump(split, f)
            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/iag_ad_01/ad/yuanweizhong/datasets/shift_val')
    parser.add_argument('--input_depth', type=str, default='/iag_ad_01/ad/yuanweizhong/datasets/shift_val')
    parser.add_argument('--output', type=str, default='data/pl_htcode/processed_datasets/shift/val_split.json')
    parser.add_argument('--tag', type=str, default='val')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)
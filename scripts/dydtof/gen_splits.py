import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')

def main(args):
    scenes = sorted(os.listdir(args.input))
    split = {'rgb_files': [],
             'depth_files': [],
             'seqs': []}
    for scene in tqdm(scenes):
        if 'zips' in scene or scene.startswith('.'):
            continue
        scene_dir = join(args.input, scene)
        seqs = sorted(os.listdir(scene_dir))
        for seq in tqdm(seqs):
            rgb_files = sorted(os.listdir(join(scene_dir, seq, 'ColorImage')))
            depth_files = sorted(os.listdir(join(scene_dir, seq, 'DepthMap')))
            assert len(rgb_files) == len(depth_files)
            start_num = len(split['rgb_files'])
            end_num = start_num + len(rgb_files)
            split['rgb_files'].extend([join(scene, seq, 'ColorImage', f) for f in rgb_files])
            split['depth_files'].extend([join(scene, seq, 'DepthMap', f) for f in depth_files])
            split['seqs'].append((start_num, end_num))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    import json 
    with open(args.output, 'w') as f:
        json.dump(split, f)
            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/bn/haotongdata/Datasets/DyDToF')
    parser.add_argument('--output', type=str, default='data/pl_htcode/processed_datasets/dydtof/split.json')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)
import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
import glob
import json
sys.path.append('.')



from lib.utils.pylogger import Log

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    return args

def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    scenes = os.listdir(args.input)
    colors, depths = [], []
    for scene in scenes:
        cams = set([item[6:12] for item in os.listdir(join(args.input, scene, 'images'))])
        for cam in cams:
            colors.extend(sorted(glob.glob(join(args.input, scene, 'images', f'scene_{cam}_final_preview',f'*.tonemap.jpg'))))
            depths.extend(sorted(glob.glob(join(args.input, scene, 'images', f'scene_{cam}_geometry_hdf5',f'*.depth_meters.hdf5'))))
            assert(len(colors) == len(depths))
    with open(args.output, 'w') as f:
        json.dump({'train_rgb_paths': colors, 'train_dpt_paths': depths,
                   'test_rgb_paths': colors[::10], 'test_dpt_paths': depths[::10]}, 
                f)
    Log.info(f'Generated {args.output}')

if __name__ == '__main__':
    args = parse_args()
    main(args)
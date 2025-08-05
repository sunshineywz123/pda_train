import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm

from lib.utils.pylogger import Log
sys.path.append('.')
import json

def main(args):
    rgb_files = []
    depth_files = []
    for seq in tqdm(sorted(os.listdir(join(args.input, 'render_data/train/img')))):
        imgs = sorted(os.listdir(join(args.input, 'render_data/train/img', seq)))
        imgs = [ img for img in imgs if '_hr' not in img]
        rgb_files.extend([join(args.input, 'render_data/train/img', seq, img) for img in imgs])
        depth_files.extend([join(args.input, 'render_data/train/depth', seq, img.replace('.jpg', '.png')) for img in imgs])
    Log.info(f'Found {len(rgb_files)} files.')
    os.makedirs(args.output, exist_ok=True)
    json.dump({'rgb_files': rgb_files, 'depth_files': depth_files}, open(join(args.output, 'train.json'), 'w'))
    Log.info(f"Saved to {args.output}")

    rgb_files = []
    depth_files = []
    for seq in tqdm(sorted(os.listdir(join(args.input, 'render_data/val/img')))):
        imgs = sorted(os.listdir(join(args.input, 'render_data/val/img', seq)))
        imgs = [ img for img in imgs if '_hr' not in img]
        rgb_files.extend([join(args.input, 'render_data/val/img', seq, img) for img in imgs])
        depth_files.extend([join(args.input, 'render_data/val/depth', seq, img.replace('.jpg', '.png')) for img in imgs])
    Log.info(f'Found {len(rgb_files)} files.')
    os.makedirs(args.output, exist_ok=True)
    json.dump({'rgb_files': rgb_files, 'depth_files': depth_files}, open(join(args.output, 'val.json'), 'w'))
    Log.info(f"Saved to {args.output}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/bn/haotongdata/Datasets/human_gps_render')
    parser.add_argument('--output', type=str, default='data/pl_htcode/processed_datasets/human_gps_render')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)
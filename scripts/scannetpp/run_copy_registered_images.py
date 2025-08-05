import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
from trdparties.colmap.read_write_model import read_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_colmap_path', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/5f99900f09/iphone/colmap_optimized/model')
    parser.add_argument('--input_rgb_path', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/5f99900f09/iphone/rgb')
    parser.add_argument('--target_path', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/5f99900f09/iphone/colmap_optimized/images')
    args = parser.parse_args()
    return args

def main(args):
    _, images, _ = read_model(args.input_colmap_path)
    for k, image in tqdm(images.items()):
        image_name = image.name
        src_path = join(args.input_rgb_path, image_name)
        tar_path = join(args.target_path, image_name)
        os.makedirs(os.path.dirname(tar_path), exist_ok=True)
        os.system('ln -s {} {}'.format(src_path, tar_path))

if __name__ == '__main__':
    args = parse_args()
    main(args)
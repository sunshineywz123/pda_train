import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm

from trdparties.colmap.read_write_model import read_images_binary, Image, write_images_binary
sys.path.append('.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/5f99900f09/merge_dslr_iphone/colmap/sparse_render_rgb_merge')
    parser.add_argument('--output', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/5f99900f09/merge_dslr_iphone/colmap/sparse_render_rgb_merge_filter')
    parser.add_argument('--filter_tag', type=str, default='render_rgb')
    args = parser.parse_args()
    return args

def main(args):
    os.makedirs(args.output, exist_ok=True)

    images = read_images_binary(join(args.input, 'images.bin'))
    new_images = {k:images[k] for k in images if args.filter_tag not in images[k].name}
    write_images_binary(new_images, join(args.output, 'images.bin'))
    os.system('cp -r {} {}'.format(join(args.input, 'cameras.bin'), join(args.output, 'cameras.bin')))
    os.system('cp -r {} {}'.format(join(args.input, 'points3D.bin'), join(args.output, 'points3D.bin')))

if __name__ == '__main__':
    args = parse_args()
    main(args)
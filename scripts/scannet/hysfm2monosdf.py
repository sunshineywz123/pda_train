import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
import glob
import cv2
sys.path.append('.')
import matplotlib.pyplot as plt
from lib.utils.dpt.eval_utils import recover_metric_depth

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/bn/haotongdata/home/linhaotong/projects/monosdf/data/scannet_store')
    parser.add_argument('--output', type=str, default='/mnt/bn/haotongdata/home/linhaotong/projects/monosdf/data/scannet/scan9')
    parser.add_argument('--scene', type=str, default='scene0050_00')
    args = parser.parse_args()
    return args

def main(args):
    depth_paths = sorted(glob.glob(join(args.input, args.scene + '*.npz')))
    tar_depth_paths = sorted(glob.glob(join(args.output, '*depth.npy')))
    # assert(len(depth_paths) == len(tar_depth_paths))
    for depth_path, tar_depth_path in zip(depth_paths, tar_depth_paths):
        depth = np.load(depth_path)['arr_0']
        depth = cv2.resize(depth, (648, 484), interpolation=cv2.INTER_LINEAR)
        center_crop_size = 384
        start_h, start_w = (484 - center_crop_size) // 2, (648 - center_crop_size) // 2
        end_h, end_w = start_h + center_crop_size, start_w + center_crop_size
        depth = depth[start_h:end_h, start_w:end_w].astype(np.float32)
        gt_depth = np.load(tar_depth_path)
        depth = recover_metric_depth(depth, gt_depth)
        depth = np.clip(depth, 0.001, 10.)
        np.save(tar_depth_path, depth)
        plt.imshow(depth)
        plt.savefig(tar_depth_path.replace('.npy', '.jpg'))
        plt.axis('off')
        plt.close()

if __name__ == '__main__':
    args = parse_args()
    main(args)
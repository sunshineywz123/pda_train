import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
import glob
import imageio
from lib.utils.vis_utils import colorize_depth_maps
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--fps', type=int, default=24)
    args = parser.parse_args()
    return args

def main(args):
    img_paths = [item for item in sorted(glob.glob(join(args.input, '*'))) if item[:1] != '.']
    rgb_paths = [img_path for img_path in img_paths if img_path[-10:] != '_depth.png' and img_path[-4:] != '.mp4']
    dpt_paths = [img_path for img_path in img_paths if img_path[-10:] == '_depth.png']
    assert(len(rgb_paths) == len(dpt_paths))
    rgbs = []
    dpts = []
    for rgb_path, dpt_path in tqdm(zip(rgb_paths, dpt_paths)):
        rgb = imageio.imread(rgb_path)
        dpt = imageio.imread(dpt_path)
        h, w, _ = rgb.shape
        rgb = rgb[:, (w // 2):, :]
        dpt = cv2.resize(dpt, (w // 2, h), interpolation=cv2.INTER_NEAREST)
        rgbs.append(rgb); dpts.append(dpt)
    dpt_min, dpt_max = np.min(dpts), np.max(dpts)
    imgs = []
    for rgb, dpt in zip(rgbs, dpts):
        dpt = (dpt - dpt_min) / (dpt_max - dpt_min)
        dpt_vis = colorize_depth_maps(dpt, 0., 1.)[0].transpose((1, 2, 0))
        img = np.concatenate([(dpt_vis*255.).astype(np.uint8), np.asarray(rgb)], axis=1)
        imgs.append(img)
    imageio.mimwrite(args.output, imgs, quality=7.5, fps=args.fps)

if __name__ == '__main__':
    args = parse_args()
    main(args)
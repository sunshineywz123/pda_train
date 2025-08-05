import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
sys.path.append('./trdparties/ZoeDepth')
from PIL import Image
import torch
import glob
from trdparties.ZoeDepth.zoedepth.models.builder import build_model
from trdparties.ZoeDepth.zoedepth.utils.config import get_config
from lib.utils.vis_utils import colorize_depth_maps
import imageio

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='path to the input folder')
    parser.add_argument('--output', type=str, help='path to output foloder')
    parser.add_argument('--zoe_type', type=str, help='path to output foloder', default='nk')
    args = parser.parse_args()
    return args


def main(args):
    
    imgs_suffix = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
    img_paths = []
    for suffix in imgs_suffix:
        img_paths.extend(glob.glob(join(args.input, '*'+suffix)))
    img_paths = sorted([img_path for img_path in img_paths if img_path[:1] != '.'])
    dpts = []
    for img_path in tqdm(img_paths):
        disp = np.load(img_path.replace('images_512x288', 'disp').replace('.png', '.npy'))
        dpts.append(disp)
    dpts = np.array(dpts)
    dpt_min, dpt_max = dpts.min(), dpts.max()
    os.makedirs(args.output, exist_ok=True)
    save_imgs = []
    for dpt, img_path in zip(dpts, img_paths):
        dpt_norm = (dpt - dpt_min) / (dpt_max - dpt_min)
        depth_vis = colorize_depth_maps(dpt_norm, 0., 1.)[0].transpose((1, 2, 0))
        save_img = np.concatenate([(depth_vis*255.).astype(np.uint8), np.asarray(Image.open(img_path).convert("RGB"))], axis=1)
        img_path = join(args.output, '{}'.format(os.path.basename(img_path)))
        imageio.imwrite(img_path, save_img)
        save_imgs.append(save_img)
    imageio.mimwrite(join(args.output, 'output.mp4'), save_imgs, fps=24, quality=8)

if __name__ == '__main__':
    args = parse_args()
    main(args)
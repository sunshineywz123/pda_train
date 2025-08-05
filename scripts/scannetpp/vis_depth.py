import os
from os.path import join
import argparse
import sys
import cv2
import numpy as np
from tqdm import tqdm

from lib.utils import vis_utils
sys.path.append('.')
import imageio


def main(args):
    # 读merge下面的RGB path
    # 读iphone下面的depth path
    # 转depth到可视化space
    rgb_names = sorted(os.listdir(join(args.input, 'merge', 'images/iphone')))
    depth_min, depth_max = 0.3, 6.0
    
    for rgb_name in tqdm(rgb_names):
        rgb_path = join(args.input, 'merge', 'images/iphone', rgb_name)
        depth_path = join(args.input, 'iphone', 'depth', rgb_name[:-4] + '.png')
        
        rgb = np.asarray(imageio.imread(rgb_path)).astype(np.uint8)
        depth = (np.asarray(imageio.imread(depth_path)) / 1000.).astype(np.float32)
        
        depth_min, depth_max = depth.min(), depth.max()
        depth = np.clip(depth, depth_min, depth_max)
        depth_vis = ((vis_utils.colorize_depth_maps(depth, depth_min, depth_max)[0].transpose(1, 2, 0)) * 255.).astype(np.uint8)
        rgb = cv2.resize(rgb, (depth_vis.shape[1], depth_vis.shape[0]), interpolation=cv2.INTER_AREA)
        vis_img = np.concatenate([rgb, depth_vis], axis=1)
        
        alpha = 0.6  # 深度图的透明度
        overlay = cv2.addWeighted(rgb, alpha, depth_vis, 1 - alpha, 0)
        # 将原始RGB、可视化的深度图和叠加图拼接在一起
        vis_img = np.concatenate([vis_img, overlay], axis=1)
        # 可视化对比两张图的align程度
        imageio.imwrite(rgb_name, vis_img)
        import ipdb; ipdb.set_trace()
    
    
    pass

def parse_args():
    # 可视化scannet pp的depth的对齐方式
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/56a0ec536c')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
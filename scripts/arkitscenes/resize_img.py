import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
import imageio
import cv2
import imageio

from lib.utils.parallel_utils import parallel_execution
sys.path.append('.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--input_dir', type=str, default='datasets/ARKitScenes/download/upsampling/Training')
    parser.add_argument('--up_scale', type=int, default=8, help='1, 2, 4, 8')
    parser.add_argument('--conf_level', type=int, default=-1, help='-1, 0, 1, 2')
    args = parser.parse_args()
    return args

def resize_img(meta):
    input_dir, name = meta
    if os.path.exists(join(input_dir, 'highres_depth_256', name)) and os.path.exists(join(input_dir, 'wide_256', name)):
        return
    highres_depth = imageio.imread(join(input_dir, 'highres_depth', name))
    wide = imageio.imread(join(input_dir, 'wide', name))
    
    highres_depth = cv2.resize(highres_depth, (256, 192), interpolation=cv2.INTER_NEAREST)
    wide = cv2.resize(wide, (256, 192), interpolation=cv2.INTER_AREA)
    
    os.makedirs(join(input_dir, 'highres_depth_256'), exist_ok=True)
    os.makedirs(join(input_dir, 'wide_256'), exist_ok=True)
    
    imageio.imwrite(join(input_dir, 'highres_depth_256', name), highres_depth)
    imageio.imwrite(join(input_dir, 'wide_256', name), wide)
    
def main(args):
    workspace = os.environ['workspace']
    scenes = sorted(os.listdir(join(workspace, args.input_dir)))
    print(scenes[-1])
    # scenes = scenes[:-1]
    
    metas = []
    for scene in scenes:
        names = os.listdir(join(workspace, args.input_dir, scene, 'wide'))
        metas.extend([(join(workspace, args.input_dir, scene), name) for name in names])
    # resize_img(metas[0])
    parallel_execution(
        metas,
        action=resize_img,
        print_progress=True,
        num_processes=128,
    )

if __name__ == '__main__':
    args = parse_args()
    main(args)
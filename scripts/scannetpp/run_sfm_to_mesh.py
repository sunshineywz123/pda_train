import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import PIL
from PIL import Image

from trdparties.colmap.read_write_model import read_model
from trdparties.colmap.read_write_model import qvec2rotmat
sys.path.append('.')

scene_to_id = {
    'cc5237fd77': 0
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/preview/scannetpp/data')
    parser.add_argument('--scene', type=str, default='cc5237fd77')
    args = parser.parse_args()
    return args

def main(args):
    input_dir = os.path.join(args.input_dir, args.scene)
    colmap_dir = join(input_dir, 'iphone', 'colmap_sfm/triangulation')
    cams, images, points3D = read_model(colmap_dir)
    import ipdb; ipdb.set_trace()
    
    pass

if __name__ == '__main__':
    args = parse_args()
    main(args)
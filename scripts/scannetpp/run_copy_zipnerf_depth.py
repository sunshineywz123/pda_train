import os
from os.path import join
import argparse
import shutil
import sys
import numpy as np
from tqdm import tqdm
import imageio
import json

from lib.utils.pylogger import Log
sys.path.append('.')
import open3d as o3d
from scipy.optimize import root
import cv2

from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from trdparties.colmap.read_write_model import read_cameras_binary, read_images_binary, read_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--colmap_path', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/56a0ec536c/merge_dslr_iphone/colmap/sparse/0')
    parser.add_argument('--zipnerf_path', type=str, default='/mnt/bn/haotongdata/home/linhaotong/projects/zipnerf-pytorch/exp/scannetpp_all_0610/56a0ec536c/test_preds')
    parser.add_argument('--output_path', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/56a0ec536c/merge_dslr_iphone/depth/v1')
    parser.add_argument('--scene', type=str, default='56a0ec536c')
    args = parser.parse_args()
    return args

def main(args):
    scene = args.scene
    colmap_path = args.colmap_path.replace('56a0ec536c', scene)
    zipnerf_path = args.zipnerf_path.replace('56a0ec536c', scene)
    output_path = args.output_path.replace('56a0ec536c', scene)
    
    cams, images, points3d = read_model(colmap_path)
    name2id = {image.name:k for k, image in images.items()}
    names = [image.name for k, image in images.items()]
    names = sorted(names)
    names = [name for name in names if 'iphone' in name]
    
    os.makedirs(join(output_path, 'vis'), exist_ok=True)
    os.makedirs(join(output_path, 'npz'), exist_ok=True)
    if not os.path.exists(join(output_path, 'info.json')):
        sys.exit(0)
    info = json.load(open(join(output_path, 'info.json'), 'r'))
    for idx, name in tqdm(enumerate(names)):
        # if idx % 10 != 0: continue
        image = images[name2id[name]]
        
        src_vis_path = join(zipnerf_path, f'distance_median_{idx:04d}.jpg')
        tar_vis_path = join(output_path, f'vis/{name[7:]}')
        shutil.copy(src_vis_path, tar_vis_path)
        
        src_npz_path = join(zipnerf_path, f'distance_median_{idx:04d}.npz')
        tar_npz_path = join(output_path, f'npz/{name[7:-4]}' + '.npz')
        depth = np.load(src_npz_path)['data']
        depth = depth * info['colmap2metric_from_zipnerf']
        depth = np.round(depth, 3)
        np.savez_compressed(tar_npz_path, data=depth)
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
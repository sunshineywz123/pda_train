import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm

from trdparties.colmap.read_write_model import Image, read_model, write_model
sys.path.append('.')

map_dict = {
    'lidar_cam01/lidar_cam01.jpg': '30.jpg',
    'lidar_cam02/lidar_cam02.jpg': '31.jpg',
    'lidar_cam03/lidar_cam03.jpg': '32.jpg',
    'lidar_cam04/lidar_cam04.jpg': '33.jpg',
} 

def main(args):
    cameras, images, points3D = read_model(args.input)
    new_images = {}
    for im_id in images:
        image = images[im_id]
        if 'lidar' in image.name:
            new_name = map_dict[image.name]
        elif 'cams24' in image.name:
            new_name = image.name[-6:]
        else:
            raise ValueError('Unknown image name: {}'.format(image.name))
        new_image = Image(
            image.id,
            image.qvec,
            image.tvec,
            image.camera_id,
            new_name,
            image.xys,
            image.point3D_ids
        )
        new_images[im_id] = new_image
    os.makedirs(args.output, exist_ok=True)
    write_model(cameras, new_images, points3D, args.output)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/bn/haotongdata/Datasets/mycapture_stray_scanner/calib5/colmap/sparse_rp/0_metric_filter')
    parser.add_argument('--output', type=str, default='/mnt/bn/haotongdata/Datasets/mycapture_stray_scanner/seq2_raw/seq2_mocap/colmap/sparse/0')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
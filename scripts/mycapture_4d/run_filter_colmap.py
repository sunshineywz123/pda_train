import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
import imageio



from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

sys.path.append('.')
from PIL import Image as PILImage
from lib.utils.pylogger import Log
from trdparties.colmap.read_write_model import Image, Point3D, read_model, write_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/data/home/linhaotong/datasets/mycapture/songyou_apartment/colmap')
    parser.add_argument('--colmap_path', type=str, default='sparse')
    parser.add_argument('--output_colmap_path', type=str, default='sparse_filter')
    args = parser.parse_args()
    return args

def main(args):
    colmap_path = join(args.input, args.colmap_path)
    output_colmap_path = join(args.input, args.output_colmap_path)
    os.makedirs(output_colmap_path, exist_ok=True)
    
    cameras, images, points3D = read_model(colmap_path)
    new_images = {}
    new_cameras = {}
    for k in images:
        if 'bgcam' in images[k].name:
            continue
        if 'for_calib' in images[k].name:
            continue
        image = images[k]
        new_image = Image(image.id, image.qvec, image.tvec, image.camera_id, os.path.basename(image.name), image.xys, image.point3D_ids)
        new_images[k] = new_image
        if new_image.camera_id not in new_cameras:
            new_cameras[new_image.camera_id] = cameras[new_image.camera_id]
    write_model(new_cameras, new_images, points3D, output_colmap_path)
    Log.info('Write metric colmap model to ' + output_colmap_path)

if __name__ == '__main__':
    args = parse_args()
    main(args)
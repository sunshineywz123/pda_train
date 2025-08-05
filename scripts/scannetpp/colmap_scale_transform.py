from email.mime import image
import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
import json

sys.path.append('.')
from lib.utils.pylogger import Log
from trdparties.colmap.read_write_model import Image, Point3D, read_model, write_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/5f99900f09/merge_dslr_iphone/colmap/sparse/0')
    parser.add_argument('--output_path', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/5f99900f09/merge_dslr_iphone/colmap/metric_model')
    parser.add_argument('--input_json_path', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/5f99900f09/merge_dslr_iphone/depth/v1/info.json')
    parser.add_argument('--scene', type=str, default='5f99900f09')
    args = parser.parse_args()
    return args

def main(args):
    input_model = args.input_path.replace('5f99900f09', args.scene)
    output_model = args.output_path.replace('5f99900f09', args.scene)
    input_json_path = args.input_json_path.replace('5f99900f09', args.scene)
    scene = args.scene

    info_json = json.load(open(input_json_path))
    colmap2metric = info_json['colmap2metric_from_zipnerf']

    cameras, images, points = read_model(input_model)
    
    os.makedirs(output_model, exist_ok=True)
    # transforms scale of colmap model
    new_images = {}
    for image_id, image in images.items():
        new_images[image_id] = Image(image_id, image.qvec, image.tvec*colmap2metric, image.camera_id, image.name, image.xys, image.point3D_ids)

    new_points = {} 
    for point_id, point in points.items():
        new_points[point_id] = Point3D(point_id, point.xyz*colmap2metric, point.rgb, point.error, point.image_ids, point.point2D_idxs)

    write_model(cameras, new_images, new_points, output_model)
    Log.info(f'Wrote metric model to {output_model}')

if __name__ == '__main__':
    args = parse_args()
    main(args)
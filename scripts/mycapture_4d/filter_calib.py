import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm

sys.path.append('.')
from trdparties.colmap.read_write_model import Image, read_model, write_model

def main(args):
    cams, images, points3d = read_model(args.input)
    new_cams = {}
    new_images = {}
    for im_id in images:
        if 'bkgd_scan' not in images[im_id].name:
            new_images[im_id] = Image(
                id=im_id,
                name=images[im_id].name[-6:],
                camera_id=images[im_id].camera_id,
                qvec=images[im_id].qvec,
                tvec=images[im_id].tvec,
                xys=images[im_id].xys,
                point3D_ids=images[im_id].point3D_ids
            )
            new_cams[images[im_id].camera_id] = cams[images[im_id].camera_id]
    os.makedirs(args.output, exist_ok=True)
    write_model(new_cams, new_images, points3d, args.output)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/data/home/linhaotong/datasets/mycapture4d/20240728_seq2/calib/colmap/sparse/0_metric')
    parser.add_argument('--output', type=str, default='/mnt/data/home/linhaotong/datasets/mycapture4d/20240728_seq2/colmap/sparse/0')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)
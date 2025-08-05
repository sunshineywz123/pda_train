import json
import os
from os.path import join
import argparse
from re import S
import sys
import numpy as np
from tqdm import tqdm
import imageio

sys.path.append('.')
from lib.utils.pylogger import Log
import open3d as o3d
from scipy.optimize import root
import cv2

from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from trdparties.colmap.read_write_model import Image, read_cameras_binary, read_images_binary, read_model, rotmat2qvec, write_cameras_text, write_images_text

def fps(c2ws, num):
    pos_all = np.array([c2w[:3, 3] for c2w in c2ws])
    quan_all = np.array([rotmat2qvec(np.linalg.inv(c2w[:3, :3])) for c2w in c2ws])

    selected_indices = [0]
    unselected_indices = list(range(1, len(pos_all)))

    for idx in range(1, num):
        if idx % 2 == 1:  # Odd index, consider position
            pos_matrix = np.linalg.norm(pos_all[unselected_indices][:, None] - pos_all[selected_indices][None, :], axis=-1) # (m, 3) - (n, 3) = (m, n, 3)
            pos_unselcted = pos_matrix.min(axis=1) # (m, n) -> (m,) 把每一个人最近的距离取出来，再取最远的。
            idx = np.argmax(pos_unselcted)
            selected_idx = unselected_indices[idx]
        else:  # Even index, consider quaternion
            # Calculate distances to all points from all selected points based on quaternion
            pos_matrix = np.linalg.norm(quan_all[unselected_indices][:, None] - quan_all[selected_indices][None, :], axis=-1) # (m, 4) - (n, 4) = (m, n, 4)
            pos_unselcted = pos_matrix.min(axis=1)
            idx = np.argmax(pos_unselcted)
            selected_idx = unselected_indices[idx]
        selected_indices.append(selected_idx)
        unselected_indices.remove(selected_idx)
    return selected_indices

def main(args):
    # cams, images, points = read_model(args.colmap_path)
    data_root  = join(args.input_path, args.scene)
    colmap_path = join(data_root, 'colmap/sparse/0_metric')
    cams, images, points3d = read_model(colmap_path)
    name2id = {image.name:k for k, image in images.items()}
    names = [image.name for k, image in images.items()]
    names = sorted(names)
    c2ws = []
    for name in names:
        image = images[name2id[name]]
        ext = np.eye(4)
        ext[:3, :3] = image.qvec2rotmat()
        ext[:3, 3] = image.tvec
        c2w = np.linalg.inv(ext)
        c2ws.append(c2w)
    selected_ids = fps(c2ws, args.fps)
    
    for split in args.splits:
        train_ids = selected_ids[:split]
        test_ids = selected_ids[args.splits[-1]:args.fps]

        files = {
            'train_ids': train_ids,
            'test_ids': test_ids,
            'train_names': [names[i] for i in train_ids],
            'test_names': [names[i] for i in test_ids],
        }

        output_colmap_path = join(data_root, 'colmap/splits/fps_split_%d' % split)
        os.makedirs(output_colmap_path, exist_ok=True)
        new_images = {im_id:Image(
            id=im_id,
            qvec=images[im_id].qvec,
            tvec=images[im_id].tvec,
            camera_id=images[im_id].camera_id,
            name=images[im_id].name,
            xys=np.empty((0, 2), dtype=np.float32),
            point3D_ids=np.empty((0,), dtype=np.int32),
        ) for im_id in images if images[im_id].name in files['train_names']}
        write_images_text(new_images, join(output_colmap_path, 'images.txt'))
        write_cameras_text(cams, join(output_colmap_path, 'cameras.txt'))
        os.system('touch {}/points3D.txt'.format(output_colmap_path))
        os.system('colmap point_triangulator --database_path {}/colmap/spsg/sfm_superpoint+superglue/database.db --image_path {}/images --input_path {}/colmap/splits/fps_split_{} --output_path {}/colmap/splits/fps_split_{} --Mapper.ba_refine_focal_length 0'.format(data_root, data_root, data_root, split, data_root, split))
        os.system('rm -rf {}/colmap/splits/fps_split_{}/*.txt'.format(data_root, split))
        output_file_path = join(data_root, 'splits', 'fps_split_%d.json' % split)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, 'w') as f:
            json.dump(files, f)
        Log.info(f"Saved to {output_file_path}")
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/mnt/data/home/linhaotong/datasets/mycapture')
    parser.add_argument('--scene', type=str, default='ede404347d')
    parser.add_argument('--fps', type=int, default=60)
    parser.add_argument('--splits', type=int, default=[7, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
import os
from os.path import join
import argparse
import sys
import imageio
import numpy as np
from sympy import N
from tqdm import tqdm

from trdparties.colmap.read_write_model import Camera, Image, read_model, write_model
sys.path.append('.')

def process_one_scene(scene):
    data_root = '/mnt/bn/haotongdata/Datasets/scannetpp/data'
    input_image_path = join(data_root, scene, 'merge_dslr_iphone', 'images')
    output_dense_path = join(data_root, scene, 'merge_dslr_iphone', 'colmap', 'dense')
    colmap_path = join(data_root, scene, 'merge_dslr_iphone', 'colmap', 'sparse_render_rgb')
    output_colmap_path = join(data_root, scene, 'merge_dslr_iphone', 'colmap', 'sparse_render_rgb_undistorted')
    os.makedirs(output_colmap_path, exist_ok=True)
    cameras, images, points3D = read_model(colmap_path)

    rgb_camera = None 
    depth_camera = None
    depth_camera_id = None

    new_cameras = {}
    for im_id in images:
        if 'iphone' in images[im_id].name:
            rgb_camera = cameras[images[im_id].camera_id]
            new_cameras[images[im_id].camera_id] = rgb_camera
        elif 'render_rgb' in images[im_id].name:
            depth_camera_id = images[im_id].camera_id
        if rgb_camera is not None and depth_camera_id is not None:
            break
    depth_camera_params = rgb_camera.params.copy()
    depth_width, depth_height = 256, 192
    depth_camera_params[[0, 2]] = depth_camera_params[[0, 2]] * depth_width / rgb_camera.width
    depth_camera_params[[1, 3]] = depth_camera_params[[1, 3]] * depth_height / rgb_camera.height
    depth_camera = Camera(
        id=depth_camera_id,
        model=rgb_camera.model,
        width=256,
        height=192,
        params=depth_camera_params
    )
    # new_cameras[depth_camera_id] = depth_camera

    new_images = {}
    for im_id in images:
        if 'render_rgb' in images[im_id].name:
            continue
            new_images[im_id] = Image(
                id=im_id,
                qvec=images[im_id].qvec,
                tvec=images[im_id].tvec,
                camera_id=depth_camera_id,
                name=images[im_id].name.replace('render_rgb', 'depth').replace('.jpg', '.png'),
                xys=np.empty((0, 2), dtype=np.float64),
                point3D_ids=np.empty((0,), dtype=np.int32)
            )
            depth = imageio.imread(join(data_root, scene, 'iphone', new_images[im_id].name.replace('render_rgb', 'depth').replace('.jpg', '.png')))
            depth = np.asarray(depth)[..., None].repeat(3, axis=-1)
            tar_path = join(data_root, scene, 'merge_dslr_iphone', 'images', new_images[im_id].name.replace('render_rgb', 'depth').replace('.jpg', '.png'))
            os.makedirs(os.path.dirname(tar_path), exist_ok=True)
            imageio.imwrite(tar_path, depth)
        else:
            new_images[im_id] = images[im_id]
    write_model(new_cameras, new_images, points3D, output_colmap_path, ext='.txt')
    cmd = f'colmap image_undistorter --image_path {input_image_path} --input_path {output_colmap_path} --output_path {output_dense_path}'
    os.system(cmd)

def main(args):
    scenes = [
    "09c1414f1b", # "31a2c91c43",
    "1ada7a0617",
    "40aec5fffa",
    "3e8bba0176",# "e7af285f7d",
    "acd95847c5",
    "578511c8a9",
    "5f99900f09",
    "c4c04e6d6c",
    "f3d64c30f8",
    "7bc286c1b6",
    "c5439f4607",
    "286b55a2bf",
    "fb5a96b1a2"]
    scenes = ['31a2c91c43', 'e7af285f7d']
    for scene in tqdm(scenes):
        process_one_scene(scene)
    # process_one_scene('5f99900f09')
    pass
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)
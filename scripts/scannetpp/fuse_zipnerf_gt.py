import json
import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
import imageio
import open3d as o3d
import open3d.core as o3c

from lib.utils.pylogger import Log
sys.path.append('.')
import open3d as o3d
from scipy.optimize import root
import cv2

from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from trdparties.colmap.read_write_model import read_cameras_binary, read_images_binary, read_model

def main(args, depth_path_dir, exp, tag):
    # cams, images, points = read_model(args.colmap_path)
    colmap_path = join(args.input_path, 'colmap/sparse_pinhole_arkit_colmap_optimized_metric')
    image_path = join(args.input_path, 'images')
    depth_path = join('/'.join(args.input_path.split('/')[:-1]), 'iphone', 'depth')
    cams, images, points3d = read_model(colmap_path)
    mesh_save_path = join(args.input_path, f'plys/zipnerf_tsdf.ply')
    os.makedirs(os.path.dirname(mesh_save_path), exist_ok=True)
    
    pcds = []
    colors = []
    
    name2id = {image.name:k for k, image in images.items()}
    names = [image.name for k, image in images.items()]
    names = sorted(names)
    names = [name for name in names if 'iphone' in name]

    voxel_size = 0.04
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.02,
        sdf_trunc=0.15,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for idx, name in tqdm(enumerate(names)):
        # if idx % 10 != 0: continue
        frame_name = os.path.basename(name)[:-4]
        image = images[name2id[name]]
        ext = np.eye(4)
        ext[:3, :3] = image.qvec2rotmat()
        ext[:3, 3] = image.tvec
        c2w = np.linalg.inv(ext)
        ixt = np.eye(3)
        cam_id = image.camera_id
        camera = cams[cam_id]
        cam_height, cam_width = camera.height, camera.width
        ixt[0, 0], ixt[1, 1], ixt[0, 2], ixt[1, 2] = camera.params[:4]
        ixt[:1] *= (args.input_w / cam_width)
        ixt[1:2] *= (args.input_h / cam_height)

        extrinsic = ext
        extrinsic = o3c.Tensor(extrinsic, o3d.core.Dtype.Float64)
        # depth_path_item = join(depth_path, f'{frame_name}.npz')
        # depth = np.load(depth_path_item)['data'] 
        # import ipdb; ipdb.set_trace()
        depth_path_item = join(depth_path_dir, f'{frame_name}.npz')
        # depth_path_item = join(depth_path, f'{frame_name}.png')
        if not os.path.exists(depth_path_item):
            continue
        depth = np.load(depth_path_item)['data']
        # depth = imageio.imread(depth_path_item) / 1000.
        # depth = cv2.resize(depth, (args.input_w, args.input_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        # depth = o3d.t.geometry.Image(depth)

        intrinsic = o3c.Tensor(ixt[:3, :3], o3d.core.Dtype.Float64)
        color_intrinsic = depth_intrinsic = intrinsic

        img = cv2.imread(join(image_path, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (args.input_w, args.input_h), interpolation=cv2.INTER_LINEAR)
        # img = img.astype(np.float32) / 255.
        # img = o3d.t.geometry.Image(img)


        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            args.input_w, args.input_h, ixt[0, 0], ixt[1, 1], ixt[0, 2], ixt[1, 2]
        )
        depth = o3d.geometry.Image(depth.astype(np.float32))
        color = o3d.geometry.Image(img.astype(np.uint8))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=args.max_depth, convert_rgb_to_intensity=False, depth_scale=1.,
        )

        volume.integrate(
            rgbd,
            camera_intrinsics,  # type: ignore
            # np.linalg.inv(ext),
            ext,
        )
    mesh = volume.extract_triangle_mesh()
    # mesh = vbg.extract_triangle_mesh().to_legacy()
    o3d.io.write_triangle_mesh(mesh_save_path, mesh)
    Log.info(f"Save mesh to {mesh_save_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/5f99900f09/merge_dslr_iphone')
    parser.add_argument('--input_h', type=int, default=756)
    parser.add_argument('--input_w', type=int, default=1008)
    parser.add_argument('--max_depth', type=float, default=5)
    parser.add_argument('--min_depth', type=float, default=0.25)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    scenes = [
    "09c1414f1b",
    "31a2c91c43",
    "1ada7a0617",
    "40aec5fffa",
    "3e8bba0176",
    "e7af285f7d",
    "acd95847c5",
    "578511c8a9",
    "5f99900f09",
    "c4c04e6d6c",
    "f3d64c30f8",
    "7bc286c1b6",
    "c5439f4607",
    "286b55a2bf",
    "fb5a96b1a2"]
    for scene in scenes:
        input_path = '/mnt/bn/haotongdata/Datasets/scannetpp/data/' + scene + '/merge_dslr_iphone'
        exp = 'june_depthanythingmetric_scannetpp_0614_hypersim_mask_far_orig_depth_anything_v2'
        depth_path_dir = f'/mnt/bn/haotongdata/Datasets/scannetpp/data/{scene}/merge_dslr_iphone/depth/v1/npz'
        tag = f'{scene}_0703'
        args.input_path = input_path
        main(args, depth_path_dir, exp, tag)
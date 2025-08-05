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

def _compute_residual_and_jacobian(x, y, xd, yd,
                                   k1=0.0, k2=0.0, k3=0.0,
                                   k4=0.0, p1=0.0, p2=0.0, ):
    """Auxiliary function of radial_and_tangential_undistort()."""
    # Adapted from https://github.com/google/nerfies/blob/main/nerfies/camera.py
    # let r(x, y) = x^2 + y^2;
    #     d(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x, y)^3 +
    #                   k4 * r(x, y)^4;
    r = x * x + y * y
    d = 1.0 + r * (k1 + r * (k2 + r * (k3 + r * k4)))

    # The perfect projection is:
    # xd = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2);
    # yd = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2);
    #
    # Let's define
    #
    # fx(x, y) = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2) - xd;
    # fy(x, y) = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2) - yd;
    #
    # We are looking for a solution that satisfies
    # fx(x, y) = fy(x, y) = 0;
    fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
    fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

    # Compute derivative of d over [x, y]
    d_r = (k1 + r * (2.0 * k2 + r * (3.0 * k3 + r * 4.0 * k4)))
    d_x = 2.0 * x * d_r
    d_y = 2.0 * y * d_r

    # Compute derivative of fx over x and y.
    fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
    fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

    # Compute derivative of fy over x and y.
    fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
    fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

    return fx, fy, fx_x, fx_y, fy_x, fy_y


def _radial_and_tangential_undistort(xd, yd, k1=0, k2=0,
                                     k3=0, k4=0, p1=0,
                                     p2=0, eps=1e-9, max_iterations=10):
    """Computes undistorted (x, y) from (xd, yd)."""
    # From https://github.com/google/nerfies/blob/main/nerfies/camera.py
    # Initialize from the distorted point.
    x = np.copy(xd)
    y = np.copy(yd)

    for _ in range(max_iterations):
        fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
            x=x, y=y, xd=xd, yd=yd, k1=k1, k2=k2, k3=k3, k4=k4, p1=p1, p2=p2)
        denominator = fy_x * fx_y - fx_x * fy_y
        x_numerator = fx * fy_y - fy * fx_y
        y_numerator = fy * fx_x - fx * fy_x
        step_x = np.where(
            np.abs(denominator) > eps, x_numerator / denominator,
            np.zeros_like(denominator))
        step_y = np.where(
            np.abs(denominator) > eps, y_numerator / denominator,
            np.zeros_like(denominator))

        x = x + step_x
        y = y + step_y

    return x, y
eps = 1e-9
max_iterations = 10
result = None
def undistort_points(fx, fy, cx, cy, k1, k2, p1, p2, x, normalized=False, denormalize=True):
    # x是Nx2的array
    global result
    if result is not None:
        return result
    x = np.atleast_2d(x)

    # put the points into normalized camera coordinates
    if not normalized:
        x = x - np.array([cx, cy]) # creates a copy
        x /= np.array([fx, fy])
    back_x = np.copy(x)
    back_fx = fx
    back_fy = fy
    xd = x[..., 0]
    yd = x[..., 1]
    
    x = np.copy(xd)
    y = np.copy(yd)
    for _ in range(max_iterations):
        fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
            x=x, y=y, xd=xd, yd=yd, k1=k1, k2=k2, k3=0, k4=0, p1=p1, p2=p2)
        denominator = fy_x * fx_y - fx_x * fy_y
        x_numerator = fx * fy_y - fy * fx_y
        y_numerator = fy * fx_x - fx * fy_x
        step_x = np.where(
            np.abs(denominator) > eps, x_numerator / denominator,
            np.zeros_like(denominator))
        step_y = np.where(
            np.abs(denominator) > eps, y_numerator / denominator,
            np.zeros_like(denominator))

        x = x + step_x
        y = y + step_y
    xu = np.stack([x, y], axis=-1)
        
    if denormalize:
        xu *= np.array([[back_fx, back_fy]])
        xu += np.array([[cx, cy]])
    result = xu
    return xu




def gen_rays(c2w, ixt, hw, k1, k2, p1, p2):
    h, w = hw
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    xy = np.stack([i, j], axis=-1).reshape(-1, 2) + 0.5
    if k1==0 and k2==0 and p1==0 and p2==0:
        xy_new = xy
    else:
        xy_new = undistort_points(ixt[0, 0], ixt[1, 1], ixt[0, 2], ixt[1, 2], k1, k2, p1, p2, xy)
    i = xy_new[..., 0].reshape(i.shape)
    j = xy_new[..., 1].reshape(j.shape)
    dirs = np.stack([(i - ixt[0, 2]) / ixt[0, 0], (j - ixt[1, 2]) / ixt[1, 1], np.ones_like(i)], axis=-1)
    dirs = np.sum(dirs[..., None, :] * c2w[:3, :3], axis=-1)
    rays_o = np.broadcast_to(c2w[:3, 3], dirs.shape)
    rays_d = dirs
    return rays_o, rays_d

def main(args):
    colmap_path = join(args.input_path, 'colmap/sparse_pinhole_arkit_colmap_optimized_metric')
    image_path = join(args.input_path, 'images')
    depth_path = join('/'.join(args.input_path.split('/')[:-1]), 'iphone', 'depth')
    cams, images, points3d = read_model(colmap_path)
    mesh_save_path = join(args.input_path, 'plys/lidar_tsdf.ply')
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
        depth_path_item = join(depth_path, f'{frame_name}.png')
        if not os.path.exists(depth_path_item):
            continue
        depth = imageio.imread(depth_path_item) / 1000.
        depth = cv2.resize(depth, (args.input_w, args.input_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
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
    parser.add_argument('--input_path', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/286b55a2bf/merge_dslr_iphone')
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
        args.input_path = input_path
        main(args)
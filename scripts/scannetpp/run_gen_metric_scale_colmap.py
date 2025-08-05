import os
from os.path import join
import argparse
import sys
import ipdb
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

from trdparties.colmap.read_write_model import Image, Point3D, read_cameras_binary, read_images_binary, read_model, write_model

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




def compute_scale(colmap_path, depth_path, conf_path=None, thresh_views=15):
    '''
    return scale from colmap to metric depth
    '''
    cams, images, points3d = read_model(colmap_path)
    name2id = {image.name:k for k, image in images.items()}
    names = [image.name for k, image in images.items()]
    names = sorted(names)

    colmap_depth, metric_depth = [], []
    for idx, name in tqdm(enumerate(names)):
        image = images[name2id[name]]
        ext = np.eye(4)
        ext[:3, :3] = image.qvec2rotmat()
        ext[:3, 3] = image.tvec
        ixt = np.eye(3)
        cam_id = image.camera_id
        camera = cams[cam_id]
        cam_height, cam_width = camera.height, camera.width
        ixt[0, 0], ixt[1, 1], ixt[0, 2], ixt[1, 2] = camera.params[:4]
        xys = image.xys[image.point3D_ids != -1]
        views = np.asarray([len(points3d[id].image_ids) for id in image.point3D_ids if id != -1])
        points = np.asarray([points3d[id].xyz for id in image.point3D_ids if id != -1])
        if len(points) == 0:
            continue
        sparse_depth = (points @ ext[:3, :3].T + ext[:3, 3:].T)[..., 2]
        uv = xys 

        if (views > thresh_views).sum() == 10:
            continue

        uv_msk = views > thresh_views
        depth_name = name.replace('.jpg', '.png').split('/')[1]
        depth = (cv2.imread(join(depth_path, depth_name), cv2.IMREAD_ANYDEPTH) / 1000.).astype(np.float32)
        uv[..., 0] *= (depth.shape[1] / cam_width)
        uv[..., 1] *= (depth.shape[0] / cam_height)
        uv = uv.astype(np.int32)
        uv_msk = uv_msk & (uv[..., 0] >= 0) & (uv[..., 0] < depth.shape[1]) & (uv[..., 1] >= 0) & (uv[..., 1] < depth.shape[0])
        uv[..., 0] = np.clip(uv[..., 0], 0, depth.shape[1] - 1)
        uv[..., 1] = np.clip(uv[..., 1], 0, depth.shape[0] - 1)
        metric_depth_item = depth[uv[:, 1], uv[:, 0]]
        uv_msk = uv_msk & (metric_depth_item < 4.)
        if conf_path is not None:
            conf = cv2.imread(join(conf_path, name), cv2.IMREAD_ANYDEPTH)
            confidence = conf[uv[:, 1], uv[:, 0]]
            uv_msk = uv_msk & (confidence == 2)

        if uv_msk.sum() > 10:
            colmap_depth.append(sparse_depth[uv_msk])
            metric_depth.append(metric_depth_item[uv_msk])
    colmap_depth = np.concatenate(colmap_depth, axis=0)
    metric_depth = np.concatenate(metric_depth, axis=0)

    degree = 1
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_model = LinearRegression(fit_intercept=False)
    ransac = RANSACRegressor(estimator=linear_model, max_trials=100000)
    model = make_pipeline(poly_features, ransac)
    model.fit(colmap_depth[:, None], metric_depth[:, None])
    a, b = model.named_steps['ransacregressor'].estimator_.coef_, model.named_steps['ransacregressor'].estimator_.intercept_
    a = a.item()
    try: b = b.item()
    except: pass
    Log.info(f'colmap2metric: {a} {b}')
    ## DEBUG
    # import matplotlib.pyplot as plt
    # rand_msk = np.random.choice(colmap_depth.shape[0], 1000)
    # rand_colmap_depth = colmap_depth[rand_msk]
    # rand_metric_depth = metric_depth[rand_msk]
    # # sorting rand metric depth
    # sort_idx = np.argsort(rand_metric_depth)
    # rand_metric_depth = rand_metric_depth[sort_idx]
    # rand_colmap_depth = rand_colmap_depth[sort_idx]
    # plt.plot(np.arange(rand_metric_depth.shape[0]), rand_metric_depth, label='metric', linewidth=2.)
    # plt.plot(np.arange(rand_colmap_depth.shape[0]), rand_colmap_depth, label='colmap', linewidth=1, linestyle='--')
    # plt.plot(np.arange(rand_colmap_depth.shape[0]), a * rand_colmap_depth + b, label='regression', linewidth=0.5, linestyle='-.')
    # plt.savefig('depth.png')
    return a
    


def process_scene(args, scene):
    scene_dir = join('/mnt/bn/haotongdata/Datasets/scannetpp/data', scene)
    root_dir = join(scene_dir, 'merge_dslr_iphone')
    colmap_path = join(root_dir, 'colmap/sparse_pinhole_arkit_colmap_optimized')
    output_colmap_path = join(root_dir, 'colmap/sparse_pinhole_arkit_colmap_optimized_metric')
    lidar_depth_path = join(scene_dir, 'iphone/depth')
    scale = compute_scale(colmap_path, lidar_depth_path)

    cameras, images, points3d = read_model(colmap_path)

    new_images = {}
    for k in images:
        image = images[k]
        new_image = Image(image.id, image.qvec, image.tvec * scale, image.camera_id, image.name, image.xys, image.point3D_ids)
        new_images[k] = new_image
    
    new_points3d = {}
    for k in points3d:
        point3d = points3d[k]
        new_point3d = Point3D(point3d.id, point3d.xyz * scale, point3d.rgb, point3d.error, point3d.image_ids, point3d.point2D_idxs)
        new_points3d[k] = new_point3d

    os.makedirs(output_colmap_path, exist_ok=True)
    write_model(cameras, new_images, new_points3d, output_colmap_path)
    Log.info('Write metric colmap model to ' + output_colmap_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/mnt/bn/haotongdata/Datasets/mycapture_stray_scanner')
    parser.add_argument('--scene', type=str, default='04042880d2')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    scenes = [
        "09c1414f1b",
        "31a2c91c43",
        "1ada7a0617",
        "5f99900f09",
        "40aec5fffa",
        "3e8bba0176",
        "e7af285f7d",
        "acd95847c5",
        "578511c8a9",
        "c4c04e6d6c",
        "f3d64c30f8",
        "7bc286c1b6",
        "c5439f4607",
        "286b55a2bf",
        "fb5a96b1a2"
    ]
    for scene in tqdm(scenes):
        process_scene(args, scene)
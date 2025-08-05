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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--colmap_path', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/56a0ec536c/merge_dslr_iphone/colmap/sparse/0')
    parser.add_argument('--lidar_depth_path', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/56a0ec536c/iphone/depth')
    parser.add_argument('--zipnerf_depth_path', type=str, default='/mnt/bn/haotongdata/home/linhaotong/projects/zipnerf-pytorch/exp/scannetpp_all_0610/56a0ec536c/test_preds')
    parser.add_argument('--output_path', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/56a0ec536c/merge_dslr_iphone/depth/v1/info.json')
    parser.add_argument('--scene', type=str, default='56a0ec536c')
    args = parser.parse_args()
    return args

def main(args):
    colmap_path = args.colmap_path.replace('56a0ec536c', args.scene)
    lidar_depth_path = args.lidar_depth_path.replace('56a0ec536c', args.scene)
    zipnerf_depth_path = args.zipnerf_depth_path.replace('56a0ec536c', args.scene)
    output_path = args.output_path.replace('56a0ec536c', args.scene)
    if os.path.exists(output_path):
        print(f'{output_path} exists')
        try: 
            print(json.load(open(output_path))['colmap2metric_from_zipnerf'])
            sys.exit(0)
        except:
            pass
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scene = args.scene
    
    cams, images, points3d = read_model(colmap_path)
    name2id = {image.name:k for k, image in images.items()}
    names = [image.name for k, image in images.items()]
    names = sorted(names)
    names = [name for name in names if 'iphone' in name]
    # np.random.seed(42) 
    # names = np.random.choice(names, 100, replace=False)
    # import ipdb; ipdb.set_trace()
    import glob
    depths = glob.glob(join(zipnerf_depth_path, '*.npz'))
    if len(names) != len(depths):
        print(f'{scene} has {len(names)} images, but {len(depths)} depth images')
        sys.exit(0)
    lowres_depths, lidar_depths, sdpts = [], [], []
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
        sparse_depth = points @ ext[:3, :3].T + ext[:3, 3:].T
        thresh_views = 15
        min_points = 200
        if np.sum(views > thresh_views) < min_points:
            Log.info(f'{name} has less than {min_points} points, with {np.sum(views > thresh_views)} points')
            continue
        lidar_depth_path_ = join(lidar_depth_path, os.path.basename(name)[:-4] + '.png')
        if not os.path.exists(lidar_depth_path_):
            continue
        lidar_depth = (imageio.imread(lidar_depth_path_)).astype(np.float32) / 1000.
        lidar_depths.append(lidar_depth)
        
        uv = xys 
        msk = views > thresh_views
        msk = np.logical_and(msk, sparse_depth[:, 2] > 0)
        msk = np.logical_and(msk, uv[:, 0] > 0)
        msk = np.logical_and(msk, uv[:, 1] > 0)
        msk = np.logical_and(msk, uv[:, 0] < cam_width - 1)
        msk = np.logical_and(msk, uv[:, 1] < cam_height - 1)
        uv = xys[msk]
        uv[:, 0] *= (lidar_depth.shape[1] / cam_width)
        uv[:, 1] *= (lidar_depth.shape[0] / cam_height)
        d = sparse_depth[:, 2][msk]
        sdpts.append(np.concatenate([uv, d[:, None]], axis=1))
        depth_path = join(zipnerf_depth_path, f'distance_median_{idx:04d}.npz')
        depth = np.load(depth_path)['data']
        depth = cv2.resize(depth, (lidar_depth.shape[1], lidar_depth.shape[0]), interpolation=cv2.INTER_NEAREST)
        lowres_depths.append(depth)
        
    # import ipdb; ipdb.set_trace()
        # depth_thresh_min = np.percentile(depth, 5.)
        # depth_thresh_max = np.percentile(depth, 80.)
        # depth_mask = np.logical_and(depth > depth_thresh_min, depth < depth_thresh_max)
        # voxel
        # depth_new_mask = np.zeros_like(depth_mask)
        # depth_new_mask[::8, ::8] = True
        # depth_mask = np.logical_and(depth_mask, depth_new_mask)
        
        
        # ixt[:1] *= (depth.shape[1] / cam_width)
        # ixt[1:2] *= (depth.shape[0] / cam_height)
        
        # rays_o, rays_d = gen_rays(c2w, ixt, depth.shape, k1, k2, p1, p2)
        
        # pcd = rays_o + rays_d * depth[..., None] 
        
        # pcds.append(pcd.reshape(-1, 3)[depth_mask.reshape(-1)])
        # colors.append(color.reshape(-1, 3)[depth_mask.reshape(-1)])
        # if idx == 10:
        #     o3d_pcd = o3d.geometry.PointCloud()
        #     o3d_pcd.points = o3d.utility.Vector3dVector(np.concatenate(pcds, 0).reshape(-1, 3))
        #     o3d_pcd.colors = o3d.utility.Vector3dVector(np.concatenate(colors, 0).reshape(-1, 3))
        #     o3d.io.write_point_cloud(f'ours.ply', o3d_pcd)
    info = {}
    sparse_depth = np.concatenate([item[:, 2] for item in sdpts], axis=0)
    lidar_depth = np.concatenate([depth[sdpt[:, 1].astype(np.int32), sdpt[:, 0].astype(np.int32)]  for sdpt, depth in zip(sdpts, lidar_depths)], axis=0)
    lowres_depth = np.concatenate([depth[sdpt[:, 1].astype(np.int32), sdpt[:, 0].astype(np.int32)]  for sdpt, depth in zip(sdpts, lowres_depths)], axis=0)
    # lowres_depth /= info['colmap2zipnerf']
    
    degree = 1
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_model = LinearRegression(fit_intercept=False)
    ransac = RANSACRegressor(estimator=linear_model, max_trials=100000)
    # ransac = RANSACRegressor(max_trials=100000)
    model = make_pipeline(poly_features, ransac)
    model.fit(sparse_depth[:, None], lidar_depth[:, None])
    a, b = model.named_steps['ransacregressor'].estimator_.coef_, model.named_steps['ransacregressor'].estimator_.intercept_
    a = a.item()
    try: b = b.item()
    except: pass
    
    colmap2metric = a,b
    Log.info(f'colmap2metric: {colmap2metric[0]}')
    
    model.fit(lowres_depth[:, None], lidar_depth[:, None])
    a, b = model.named_steps['ransacregressor'].estimator_.coef_, model.named_steps['ransacregressor'].estimator_.intercept_
    a = a.item()
    try: b = b.item()
    except: pass
    
    colmap2metric_from_zipnerf = a,b
    Log.info(f'colmap2metric_from_zipnerf: {colmap2metric_from_zipnerf[0]}')
    info['colmap2metric'] = colmap2metric[0]
    info['colmap2metric_from_zipnerf'] = colmap2metric_from_zipnerf[0]
    with open(output_path, 'w') as f:
        json.dump(info, f, indent=4)
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
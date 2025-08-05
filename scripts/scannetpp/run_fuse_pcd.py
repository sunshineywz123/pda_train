import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
import imageio
sys.path.append('.')
import open3d as o3d
from scipy.optimize import root
import cv2

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
    parser.add_argument('--pred_path', type=str, default='/mnt/bn/haotongdata/home/linhaotong/projects/zipnerf-pytorch/exp/scannetpp_new/56a0ec536c_both_optim_all/test_preds')
    parser.add_argument('--depth_type', type=str, default='distance_median')
    parser.add_argument('--colmap_path', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/56a0ec536c/merge/colmap/optimized')
    parser.add_argument('--gt_path', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/56a0ec536c/merge/images')
    parser.add_argument('--normalized_scale_factor', type=float, default=1.0110274022579326)   
    parser.add_argument('--output_ply_path', type=str, default='output.ply')
    args = parser.parse_args()
    return args

def gen_rays(c2w, ixt, hw, k1, k2, p1, p2):
    h, w = hw
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    xy = np.stack([i, j], axis=-1).reshape(-1, 2)
    xy_new = undistort_points(ixt[0, 0], ixt[1, 1], ixt[0, 2], ixt[1, 2], k1, k2, p1, p2, xy)
    i = xy_new[..., 0].reshape(i.shape)
    j = xy_new[..., 1].reshape(j.shape)
    dirs = np.stack([(i - ixt[0, 2]) / ixt[0, 0], (j - ixt[1, 2]) / ixt[1, 1], np.ones_like(i)], axis=-1)
    dirs = np.sum(dirs[..., None, :] * c2w[:3, :3], axis=-1)
    rays_o = np.broadcast_to(c2w[:3, 3], dirs.shape)
    rays_d = dirs
    return rays_o, rays_d

def main(args):
    # cams, images, points = read_model(args.colmap_path)
    cams, images = read_cameras_binary(join(args.colmap_path, 'cameras.bin')), read_images_binary(join(args.colmap_path, 'images.bin'))
    pcds = []
    colors = []
    
    name2id = {image.name:k for k, image in images.items()}
    names = [image.name for k, image in images.items()]
    names = sorted(names)
    names = [name for name in names if 'iphone' in name]
    # import ipdb; ipdb.set_trace()
    
    # color_images = sorted(os.listdir(args.gt_path))
    for idx, name in tqdm(enumerate(names)):
        # if idx % 10 != 0: continue
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
        if camera.model == 'OPENCV': k1, k2, p1, p2 = camera.params[4:]
        else: k1, k2, p1, p2 = 0., 0., 0., 0.
        # depth_path = join(args.pred_path, f'Meetingroom-frame-{idx+1:06d}_depth_align_ransac_sfm.npz')
        # depth = np.load(depth_path)['arr_0']
        depth_path = join(args.pred_path, f'{args.depth_type}_{idx:04d}.npz')
        depth = np.load(depth_path)['data'] / args.normalized_scale_factor
        color_path = join(args.gt_path, name)
        color = np.asarray(imageio.imread(color_path) / 255.).astype(np.float32)
        
        depth_thresh_min = np.percentile(depth, 5.)
        depth_thresh_max = np.percentile(depth, 80.)
        depth_mask = np.logical_and(depth > depth_thresh_min, depth < depth_thresh_max)
        # voxel
        depth_new_mask = np.zeros_like(depth_mask)
        depth_new_mask[::8, ::8] = True
        depth_mask = np.logical_and(depth_mask, depth_new_mask)
        
        
        # ixt[:1] *= (depth.shape[1] / cam_width)
        # ixt[1:2] *= (depth.shape[0] / cam_height)
        
        rays_o, rays_d = gen_rays(c2w, ixt, depth.shape, k1, k2, p1, p2)
        
        pcd = rays_o + rays_d * depth[..., None] 
        
        pcds.append(pcd.reshape(-1, 3)[depth_mask.reshape(-1)])
        colors.append(color.reshape(-1, 3)[depth_mask.reshape(-1)])
        # if idx == 10:
        #     o3d_pcd = o3d.geometry.PointCloud()
        #     o3d_pcd.points = o3d.utility.Vector3dVector(np.concatenate(pcds, 0).reshape(-1, 3))
        #     o3d_pcd.colors = o3d.utility.Vector3dVector(np.concatenate(colors, 0).reshape(-1, 3))
        #     o3d.io.write_point_cloud(f'ours.ply', o3d_pcd)
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(np.concatenate(pcds, 0).reshape(-1, 3))
    o3d_pcd.colors = o3d.utility.Vector3dVector(np.concatenate(colors, 0).reshape(-1, 3))
    o3d.io.write_point_cloud(args.output_ply_path, o3d_pcd)

if __name__ == '__main__':
    args = parse_args()
    main(args)
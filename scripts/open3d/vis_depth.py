import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
import imageio
import open3d as o3d

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dpt_path', type=str)
    parser.add_argument('--input_rgb_path', type=str, default=None)
    parser.add_argument('--depth_type', type=str, default='min_max')
    parser.add_argument('--scale', type=float, default=1.)
    parser.add_argument('--shift', type=float, default=0.)
    parser.add_argument('--focal', type=float, default=543.) # 868 * 0.625, 768 -> 480
    parser.add_argument('--filter_min', type=float, default=0.) # 868 * 0.625, 768 -> 480
    parser.add_argument('--filter_max', type=float, default=100.) # 868 * 0.625, 768 -> 480
    args = parser.parse_args()
    return args

def unproject(dpt, focal):
    h, w = dpt.shape
    cx, cy = w / 2, h / 2
    ixt = np.asarray([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])
    x, y = np.arange(w), np.arange(h)
    xx, yy = np.meshgrid(x, y)
    xyz = np.stack((xx, yy, np.ones_like(xx)), axis=-1).astype(np.float32)
    xyz *= dpt[..., None]
    point_cloud = xyz @ np.linalg.inv(ixt.T)
    return point_cloud

def read_depth(args):
    if args.input_dpt_path[-3:] == 'npz':
        # hypersim gt depth
        dpt = np.asarray(np.load(args.input_dpt_path)['data'])
    elif args.input_dpt_path[-3:] in ['npy', 'npz']:
        import ipdb; ipdb.set_trace()
    else:
        dpt = imageio.imread(args.input_dpt_path)
        if dpt.dtype == np.uint8: dpt = dpt.astype(np.float32) / 255.
    print(dpt.min(), dpt.max())
    if len(dpt.shape) == 3: dpt = dpt[..., 0]
    if args.depth_type == 'log':
        dpt = np.exp(dpt * (np.log(80.) - np.log(0.5)) + np.log(0.5))
    elif args.depth_type == 'disp':
        dpt = np.clip(dpt, 1/80., None)
        dpt = 1/dpt
    else:
        pass
    dpt = dpt * args.scale + args.shift
    min_val = np.percentile(dpt, args.filter_min)
    max_val = np.percentile(dpt, args.filter_max)
    msk = np.logical_and(dpt>min_val, dpt<max_val)
    return dpt, msk
    

def main(args):
    dpt, msk = read_depth(args)
    points = unproject(dpt, args.focal)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, 3)[msk.reshape(-1)])
    if args.input_rgb_path is not None:
        img = imageio.imread(args.input_rgb_path)
        if img.shape[1] == 2 * dpt.shape[1]:
            img = img[:, dpt.shape[1]:]
        pcd.colors = o3d.utility.Vector3dVector(img.reshape(-1, 3)[msk.reshape(-1)] / 255.)
    cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    o3d.visualization.draw_geometries([pcd, cam])
    
if __name__ == '__main__':
    args = parse_args()
    main(args)

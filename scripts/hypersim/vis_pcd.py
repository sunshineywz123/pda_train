import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
import h5py
import imageio
import trimesh
import open3d as o3d
from lib.utils.pylogger import Log

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/mnt/remote/D002/Datasets/hypersim_singlescene_full/ai_001_001')
    parser.add_argument('--cam_path', type=str, default='cam_00')
    args = parser.parse_args()
    return args

def main(args):
    geometry_path = join(args.data_root, 'images', 'scene_{}_geometry_hdf5'.format(args.cam_path))
    position_frames = sorted([join(geometry_path, f) \
                               for f in os.listdir(geometry_path) \
                               if 'position' in f])
    
    rgb_path = join(args.data_root, 'images', 'scene_{}_final_preview'.format(args.cam_path))
    rgb_frames = sorted([join(rgb_path, f) \
                         for f in os.listdir(rgb_path) \
                         if 'color.jpg' in f])
    pcds = np.asarray([np.asarray(h5py.File(frame)['dataset']) for frame in position_frames]).reshape(-1, 3)
    rgbs = np.asarray([np.asarray(imageio.imread(rgb)) for rgb in rgb_frames]).reshape(-1, 3)
    
    # voxelize, voxel_size = 0.01
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcds)
    pcd.colors = o3d.utility.Vector3dVector(rgbs / 255.)
    voxel_size = 0.1
    pcd = pcd.voxel_down_sample(voxel_size)
    pcds = np.asarray(pcd.points)
    rgbs = np.asarray(pcd.colors) * 255
    
    mesh = trimesh.Trimesh(vertices=pcds, vertex_colors=rgbs)
    output_path = join(args.data_root, 'images', 'scene_{}_mesh.ply'.format(args.cam_path))
    mesh.export(output_path)

if __name__ == '__main__':
    args = parse_args()
    main(args)
import os
from os.path import join
import argparse
import sys
import ipdb
import numpy as np
from tqdm import tqdm
sys.path.append('.')
import open3d as o3d
import torch
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/5f99900f09/merge_dslr_iphone/depth/v1/pcd/points.ply')
    parser.add_argument('--input_transform', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/5f99900f09/merge_dslr_iphone/transforms.json')
    parser.add_argument('--output_path', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/5f99900f09/merge_dslr_iphone/dense_pc_zipnerf.ply')
    args = parser.parse_args()
    return args

def main(args):
    pcd = o3d.io.read_point_cloud(args.input_path)
    points = torch.from_numpy(np.asarray(pcd.points))
    points3D_rgb = (np.asarray(pcd.colors) * 255.).astype(np.uint8)
    transforms = json.load(open(args.input_transform))['applied_transform']
    applied_transform = torch.from_numpy(np.asarray(transforms))
    
    points3D = torch.einsum("ij,bj->bi", applied_transform[:3, :3], points) + applied_transform[:3, 3]
    
    with open(args.output_path, "w") as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points3D)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uint8 red\n")
        f.write("property uint8 green\n")
        f.write("property uint8 blue\n")
        f.write("end_header\n")
        
        for coord, color in tqdm(zip(points3D, points3D_rgb)):
            x, y, z = coord
            r, g, b = color
            f.write(f"{x:8f} {y:8f} {z:8f} {r} {g} {b}\n")

if __name__ == '__main__':
    args = parse_args()
    main(args)
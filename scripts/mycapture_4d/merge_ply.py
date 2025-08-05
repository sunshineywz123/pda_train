import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
from plyfile import PlyData, PlyElement

def main(args):
    ply1 = PlyData.read(args.input)
    ply2 = PlyData.read(args.input2)
    vertices = np.concatenate((ply1['vertex'].data, ply2['vertex'].data))
    merged_element = PlyElement.describe(vertices, 'vertex')
    merged_ply = PlyData([merged_element], text=False)
    merged_ply.write(args.output)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/data/home/linhaotong/datasets/mycapture4d/20240728_seq1/calib/frames/000000/dense/sparse/points3D.ply')
    parser.add_argument('--input2', type=str, default='/mnt/data/home/linhaotong/datasets/mycapture4d/20240728_seq1/pcd_init/lidar_depth/000000.ply')
    parser.add_argument('--output', type=str, default='/mnt/data/home/linhaotong/datasets/mycapture4d/20240728_seq1/calib/frames/000000/dense/sparse/points3D.ply')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)
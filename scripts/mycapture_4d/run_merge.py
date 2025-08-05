import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')

import numpy as np
from plyfile import PlyData, PlyElement

def merge_ply_files(ply_path1, ply_path2, output_path):
    # 读取两个PLY文件
    ply1 = PlyData.read(ply_path1)
    ply2 = PlyData.read(ply_path2)
    
    # 确保两个PLY文件的元素具有相同的属性
    # assert ply1.elements[0].properties == ply2.elements[0].properties, "PLY files must have the same properties"
    
    # 合并点云数据
    points1 = np.stack((
        np.asarray(ply1.elements[0]["x"]),
        np.asarray(ply1.elements[0]["y"]),
        np.asarray(ply1.elements[0]["z"])
    ), axis=1)
    
    points2 = np.stack((
        np.asarray(ply2.elements[0]["x"]),
        np.asarray(ply2.elements[0]["y"]),
        np.asarray(ply2.elements[0]["z"])
    ), axis=1)
    
    # 合并点云
    merged_points = np.vstack((points1, points2))
    
    # 创建新的属性数组
    merged_properties = []
    for prop in ply1.elements[0].properties:
        merged_properties.append(prop.name)
    
    # 将合并后的点云数据保存为一个新的PLY文件
    vertex_all = np.empty(len(merged_points), dtype=merged_properties)
    for i, prop in enumerate(ply1.elements[0].properties):
        vertex_all[prop.name] = merged_points[:, i] if prop.name in ['x', 'y', 'z'] else np.zeros(len(merged_points))
    
    # 填充其他属性
    for prop in ply1.elements[0].properties:
        if prop.name not in ['x', 'y', 'z']:
            if prop.name in ply1.elements[0].data.dtype.names:
                vertex_all[prop.name] = np.concatenate((
                    np.asarray(ply1.elements[0][prop.name]),
                    np.asarray(ply2.elements[0][prop.name])
                ))
            else:
                vertex_all[prop.name] = np.zeros(len(merged_points))
    
    element = PlyElement.describe(vertex_all, 'vertex')
    PlyData([element]).write(output_path)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input1', type=str, default='/mnt/bn/haotongdata/home/linhaotong/projects/2dgs/output/mycapture_4d_calib_rp_frame1_depth_7frames/point_cloud/iteration_4000/point_cloud.ply')
    parser.add_argument('--input2', type=str, default='/mnt/bn/haotongdata/home/linhaotong/projects/2dgs/output/mycapture_4d_calib_rp_metric/point_cloud/iteration_30000/point_cloud.ply')
    parser.add_argument('--output', type=str, default='output.ply')
    args = parser.parse_args()
    return args

def main(args):
    merge_ply_files(args.input1, args.input2, args.output)

if __name__ == '__main__':
    args = parse_args()
    main(args)
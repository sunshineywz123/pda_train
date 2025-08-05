import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
import pandas as pd
import h5py
import cv2

from lib.utils.parallel_utils import parallel_execution


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    return args


def get_intrinsic(scene, dfs):
    # Create the placeholder for the intrinsic matrix
    K = np.eye(3)

    # Find the camera parameters for the scene
    df = dfs.loc[scene]
    W, H = int(df["settings_output_img_width"]), int(df["settings_output_img_height"])
    M = np.array([[df["M_proj_00"], df["M_proj_01"], df["M_proj_02"], df["M_proj_03"]],
                  [df["M_proj_10"], df["M_proj_11"], df["M_proj_12"], df["M_proj_13"]],
                  [df["M_proj_20"], df["M_proj_21"], df["M_proj_22"], df["M_proj_23"]],
                  [df["M_proj_30"], df["M_proj_31"], df["M_proj_32"], df["M_proj_33"]]])

    # Fill the intrinsic matrix
    K[0, 0] = M[0, 0] * W / 2
    K[0, 2] = -(M[0, 2] * W - W) / 2
    K[1, 1] = M[1, 1] * H / 2
    K[1, 2] = (M[1, 2] * H + H) / 2

    return K, H, W


def convert_distance_to_depth(distance, H, W, focal):
    # Convert the distance map to the depth map
    x = np.linspace((-0.5 * W) + 0.5, (0.5 * W) - 0.5, W).reshape(1, W).repeat(H, 0).astype(np.float32)[:, :, None]
    y = np.linspace((-0.5 * H) + 0.5, (0.5 * H) - 0.5, H).reshape(H, 1).repeat(W, 1).astype(np.float32)[:, :, None]
    z = np.full([H, W, 1], focal, np.float32)
    plane = np.concatenate([x, y, z], 2)  # (H, W, 3)
    depth = distance / np.linalg.norm(plane, 2, 2) * focal  # (H, W)

    return depth


# Define some global variables for hypersim
hypersim_cam_dir = '_detail/cam_{subscene}'
hypersim_img_dir = 'images/scene_cam_{subscene}_final_preview'
hypersim_dpt_dir = 'images/scene_cam_{subscene}_geometry_hdf5'
hypersim_img_pattern = 'frame.{frame:04d}.tonemap.jpg'
hypersim_dpt_pattern = 'frame.{frame:04d}.depth_meters.hdf5'


hypersim_meta_dir = 'data/pl_htcode/processed_datasets/HyperSim'
hypersim_meta_camera = 'metadata_camera_parameters.csv'
hypersim_dir = 'data/pl_htcode/datasets/HyperSim/all'

meta_camera = pd.read_csv(join(hypersim_meta_dir, hypersim_meta_camera), index_col="scene_name")

def process_one_scene(scene):
    # 获取metacamera
    ixt, h, w = get_intrinsic(scene, meta_camera)
    
    # 获取subscenes
    subdirs = sorted(os.listdir(join(hypersim_dir, scene, 'images')))
    subscenes = []
    for subdir in subdirs:
        subscene = '_'.join(subdir.split('_')[1:3])
        if subscene not in subscenes:
            subscenes.append(subscene)
    
    def process_subscene(subscene):
        dpts = sorted(os.listdir(join(hypersim_dir, 
                                      scene, 
                                      'images',
                                      f'scene_{subscene}_geometry_hdf5')))
        dpts = [dpt for dpt in dpts if dpt.endswith('depth_meters.hdf5')]
        save_dir = join(hypersim_meta_dir, 'processed_data', scene, 'images', f'scene_{subscene}_geometry_hdf5')
        os.makedirs(save_dir, exist_ok=True)
        
        for dpt in dpts:
            dpt_path = join(hypersim_dir, 
                            scene, 
                            'images',
                            f'scene_{subscene}_geometry_hdf5',
                            dpt)
            with h5py.File(dpt_path, "r") as f: distance = f['dataset'][:]
            if np.isinf(distance).sum() > 0:
                import ipdb; ipdb.set_trace()
            depth = convert_distance_to_depth(distance, h, w, ixt[0, 0])
            # depth = (depth * 1000).astype(np.uint16)
            # save_path = join(save_dir, 
            #                  dpt.replace('depth_meters.hdf5', 'depth.png'))
            # cv2.imwrite(save_path, depth)
            depth = np.round(depth, 3)
            save_path = join(save_dir, 
                             dpt.replace('depth_meters.hdf5', 'depth.npz'))
            np.savez_compressed(save_path, data=depth)
            
        
    for subscene in subscenes:
        process_subscene(subscene)
    
def main(args):
    # 计算depth， 保存depth
    scenes = sorted(os.listdir(hypersim_dir))
    parallel_execution(scenes, 
                       action=process_one_scene, 
                       num_processes=8, 
                       print_progress=True)
    # for scene in scenes:
    #     process_one_scene(scene)

if __name__ == '__main__':
    args = parse_args()
    main(args)
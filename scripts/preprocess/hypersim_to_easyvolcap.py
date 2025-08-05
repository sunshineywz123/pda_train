# This script converts the Hypersim dataset to EasyVolcap dataset format, mainly does the following things:
# 1. Take the tilt-shift photography parameters into consideration and get the accurate intrinsic matrix
# 2. Convert the camera parameters to the EasyVolcap format
# 3. Convert the distance map to the depth map and save the depth maps as `.exr` files
# 4. Link the images directly to the output directory
# 5. Organize the whole dataset according to the official split file

# https://github.com/apple/ml-hypersim

import os
import cv2
import h5py
import argparse
import numpy as np
import pandas as pd
from glob import glob
from os.path import join, dirname
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.data_utils import save_image
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.easy_utils import read_camera, write_camera


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

# Define some global variables for EasyVolcap
easyvolcap_img_dir = 'images'
easyvolcap_dpt_dir = 'depths'


@catch_throw
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hypersim_root', type=str, default='data/workspace/hypersim')
    parser.add_argument('--easyvolcap_root', type=str, default='data/workspace/hypersim_evc')
    parser.add_argument('--scene', type=str, default=None)
    parser.add_argument('--ext', type=str, default='.jpg')
    parser.add_argument('--white_thres', type=float, default=0.2)
    args = parser.parse_args()

    # Global variables
    hypersim_root = args.hypersim_root
    easyvolcap_root = args.easyvolcap_root

    # Load the processed perspective projection matrix for all scenes to get the intrinsic matrix
    params = pd.read_csv(join(hypersim_root, 'metadata_camera_parameters.csv'), index_col="scene_name")
    # Load the offical split file
    splits = pd.read_csv(join(hypersim_root, 'metadata_images_split_scene_v1.csv'))

    # Error list
    error_list = []

    def process_scene(scene: str):
        # Find all subdirectories under the scene directory
        subscenes = [basename(p).split('_')[-1] for p in glob(join(hypersim_root, scene, '_detail', 'cam_**'), recursive=True)]
        subscenes = sorted(subscenes)

        # Filter the split file with key `scene_name`, and drop nan for better robustness
        scene_splits = splits[splits['scene_name'] == scene].dropna(subset='split_partition_name')

        def process_subscene(subscene: str):
            # Determine the split according to the offical split file
            split = scene_splits[scene_splits['camera_name'] == f'cam_{subscene}'].iloc[0]['split_partition_name']
            if split not in ['train', 'val', 'test']: return

            # The input easyvolcap format data directory is for sure
            subscene_in_cam_dir = join(hypersim_root, scene, hypersim_cam_dir.format(subscene=subscene))
            subscene_in_img_dir = join(hypersim_root, scene, hypersim_img_dir.format(subscene=subscene))
            subscene_in_dpt_dir = join(hypersim_root, scene, hypersim_dpt_dir.format(subscene=subscene))
            # The output easyvolcap format data directory is for sure
            subscene_out_dir = join(easyvolcap_root, split, scene, subscene)

            # Return if the output directory already exists
            if exists(subscene_out_dir): return

            # Load the c2w rotation matrices and translation vectors
            with h5py.File(join(subscene_in_cam_dir, 'camera_keyframe_orientations.hdf5'), "r") as f: c2wRs = f['dataset'][:]  # (N, 3, 3)
            with h5py.File(join(subscene_in_cam_dir, 'camera_keyframe_positions.hdf5'), "r") as f: c2wTs = f['dataset'][:]  # (N, 3)
            # Convert the c2w rotation matrices from OpenGL to OpenCV
            c2wRs[..., 1:3] = -c2wRs[..., 1:3]  # (N, 3, 3)
            # Convert the c2w translation vectors unit from asset unit to meters
            metas = pd.read_csv(join(hypersim_root, scene, '_detail', 'metadata_scene.csv'))
            scale = metas[metas['parameter_name'] == 'meters_per_asset_unit']['parameter_value'].values
            c2wTs = c2wTs * scale  # (N, 3)
            # Combine the c2w rotation matrices and translation vectors
            c2ws = np.concatenate([c2wRs[:, :, :3], c2wTs[:, :, None]], axis=2)  # (N, 3, 4)

            # Compute the camera intrinsics
            K, H, W = get_intrinsic(scene, params)

            cameras = dotdict()
            # Sorted image names and distance map names
            ims = sorted([i for i in os.listdir(subscene_in_img_dir) if 'tonemap' in i])
            dis = sorted([d for d in os.listdir(subscene_in_dpt_dir) if d.endswith('.hdf5')])

            # Some scenes may have missing images or distance maps
            if len(ims) != len(c2ws): # The number of images and cameras should be the same
                log(yellow(f'Wanring: number of images and cameras do not match: {len(ims)} != {len(c2ws)}'))

            # Initialize the counter
            cnt = 0
            # Process the left view and right view respectively
            for i, c2w in enumerate(c2ws):
                # Check whether the index `i` has a valid image and distance map
                if hypersim_img_pattern.format(frame=i) not in ims or hypersim_dpt_pattern.format(frame=i) not in dis:
                    continue

                # Check the correspondence between the image and the distance map
                if f'{i:04d}' not in ims[cnt] or f'{i:04d}' not in dis[cnt]:
                    log(red(f'Error: image and distance map do not match: {ims[cnt]} != {dis[cnt]}'))
                    error_list.append((scene, subscene, ims[cnt], dis[cnt]))
                    return

                # Camera data
                cameras[f'{cnt:06d}'] = dotdict()  # might have more than 100 cameras?
                cam = cameras[f'{cnt:06d}']
                cam.H = H
                cam.W = W
                cam.K = K
                cam.R = c2w[:3, :3].T  # (3, 3)
                cam.T = -cam.R @ c2w[:3, 3]  # (3, 3) @ (3,)
                cam.D = np.zeros((1, 5))

                # Create the output directory for image and depth
                os.makedirs(join(subscene_out_dir, easyvolcap_img_dir, f'{cnt:06d}'), exist_ok=True)
                # Link the image to the output directory
                os.system(f"ln -s {os.path.abspath(join(subscene_in_img_dir, ims[cnt]))} {join(subscene_out_dir, easyvolcap_img_dir, f'{cnt:06d}', f'{0:06d}{args.ext}')}")

                # Create the output directory for depth
                os.makedirs(join(subscene_out_dir, easyvolcap_dpt_dir, f'{cnt:06d}'), exist_ok=True)
                # Load the original distance map and convert it to the depth map
                with h5py.File(join(subscene_in_dpt_dir, dis[cnt]), "r") as f: dis_map = f['dataset'][:]  # (H, W)
                dpt_map = convert_distance_to_depth(dis_map, H, W, K[0, 0])
                # Save the converted depth map as `.exr`
                save_image(join(subscene_out_dir, easyvolcap_dpt_dir, f'{cnt:06d}', f'{0:06d}.exr'), dpt_map[..., None])

                # Update the counter
                cnt = cnt + 1

            # Write the camera data
            write_camera(cameras, subscene_out_dir)  # extri.yml and intri.yml

        # Process each subscene parallel
        parallel_execution(subscenes, action=process_subscene, sequential=False, print_progress=True)

    # Find and process all scenes
    scenes = sorted(os.listdir(hypersim_root))
    if args.scene is not None:
        log(yellow(f'Only process scene: {args.scene}'))
        scenes = [x for x in scenes if args.scene in x]
    parallel_execution(scenes, action=process_scene, sequential=True, print_progress=True)  # literally a for-loop

    # Log the error list
    if len(error_list) > 0:
        log(red(f'Error list: {len(error_list)}'))
        for e in error_list:
            log(red(f'{e}'))


if __name__ == '__main__':
    main()

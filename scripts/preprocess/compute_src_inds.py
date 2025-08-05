# This script computes the source indices using the ground truth depth map and camera parameters
# Assuming multi-view format dataset, not monocular dataset

import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
from glob import glob
from os.path import join, dirname
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.depth_utils import depth2xyz
from easyvolcap.utils.data_utils import save_image, load_depth
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.easy_utils import read_camera, write_camera
from easyvolcap.utils.cam_utils import compute_relative_angle, compute_relative_dists
from easyvolcap.utils.math_utils import affine_inverse, affine_padding, torch_inverse_3x3, point_padding


def load_cameras(data_root, intri_file='intri.yml', extri_file='extri.yml'):
    cameras = read_camera(join(data_root, intri_file), join(data_root, extri_file))
    camera_names = np.asarray(sorted(list(cameras.keys())))  # NOTE: sorting camera names
    cameras = dotdict({k: cameras[k] for k in camera_names})

    # NOTE: ALWAYS, ALWAYS, SORT CAMERA NAMES.
    Hs = torch.as_tensor([cameras[k].H for k in camera_names], dtype=torch.float)  # (V,)
    Ws = torch.as_tensor([cameras[k].W for k in camera_names], dtype=torch.float)  # (V,)
    Ks = torch.as_tensor([cameras[k].K for k in camera_names], dtype=torch.float)  # (V, 3, 3)
    Rs = torch.as_tensor([cameras[k].R for k in camera_names], dtype=torch.float)  # (V, 3, 3)
    Ts = torch.as_tensor([cameras[k].T for k in camera_names], dtype=torch.float)  # (V, 3, 1)
    Cs = -Rs.mT @ Ts  # (V, 3, 1)
    w2cs = torch.cat([Rs, Ts], dim=-1)  # (V, 3, 4)
    c2ws = affine_inverse(w2cs)  # (V, 3, 4)

    return camera_names, Hs, Ws, Ks, Rs, Ts, Cs, w2cs, c2ws


@catch_throw
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/workspace/hypersim_evc')
    parser.add_argument('--depths_dir', type=str, default='depths')
    parser.add_argument('--source_dir', type=str, default='source')
    parser.add_argument('--H', type=int, default=None)
    parser.add_argument('--W', type=int, default=None)
    parser.add_argument('--threshold_proj', type=float, default=0.75)
    parser.add_argument('--threshold_dist', type=float, default=2.00)
    parser.add_argument('--threshold_rots', type=float, default=30.)  # Do not use this for now
    parser.add_argument('--scene', type=str, default=None)
    args = parser.parse_args()

    # Global variables
    data_root = args.data_root
    depths_dir = args.depths_dir
    source_dir = args.source_dir

    def process_scene(scene: str):
        # Load camera parameters for the scene
        camera_names, Hs, Ws, Ks, Rs, Ts, Cs, w2cs, c2ws = load_cameras(scene)
        V = len(camera_names)

        # Create the source similarity and index for the current scene
        scene_sims = torch.zeros((V, V))  # (V, V)
        scene_inds = torch.zeros((V, V), dtype=torch.int)  # (V, V)

        # Save the source similarity and index
        output_dir = join(scene, source_dir)
        # Skip if the file already exist
        if exists(join(output_dir, 'src_inds.npy')):
            log(green(f'Skip scene: {scene}'))
            return
        # Create the output directory
        os.makedirs(output_dir, exist_ok=True)

        def process_view(index: int, view: str):
            # Load the depth map of the target view
            tar_dpt_dir = join(scene, depths_dir, view)
            tar_dpt = load_depth(join(tar_dpt_dir, sorted(os.listdir(tar_dpt_dir))[0]))[..., 0]  # (H, W)
            tar_dpt = torch.as_tensor(tar_dpt)  # (H, W)

            # Get the target view's extrinsic and intrinsic matrix
            tar_ext, src_exts = w2cs[index], w2cs  # (3, 4), (V, 3, 4)
            tar_ixt, src_ixts = Ks[index], Ks  # (3, 3), (V, 3, 3)
            # Get the image height and width, NOTE: assume all views have the same image resolution
            H, W = int(Hs[0]), int(Ws[0])
            H = args.H if args.H is not None else H
            W = args.W if args.W is not None else W
            if H < 0 or W < 0: raise ValueError(f"Invalid resolution: {H}x{W}")

            # Back project the target view depth to world xyz
            tar_xyz = depth2xyz(tar_dpt[None, None], tar_ixt[None, None], tar_ext[None, None])[0, 0]  # (3, H, W)
            # Project the world xyz to other source views
            src_xyzs = src_ixts @ (src_exts[..., :3, :3] @ tar_xyz.reshape(1, 3, -1) + src_exts[..., :3, 3:])  # (V, 3, H*W)
            src_grid = src_xyzs / src_xyzs[..., -1:, :].clip(1e-6)  # (V, 3, H*W)
            src_grid = src_grid[..., :2, :].reshape(V, 2, H, W)  # (V, 2, H, W)
            # Compute the inside image pixel percentage of each source view
            src_msks = (src_grid[:, :1] >= 0) * (src_grid[:, :1] < W) * (src_grid[:, 1:] >= 0) * (src_grid[:, 1:] < H)  # (V, 1, H, W)
            src_cnts = torch.sum(src_msks, dim=(-3, -2, -1))  # (V,)
            # Projects similarity
            sims_proj = src_cnts / (H * W)  # (V,)
            # sims_projects = sims_projects / torch.norm(sims_projects, dim=0)  # (V,)
            sims_proj[index] = 1e5  # NOTE: set the target view similarity to a large value for sorting
            src_sims, src_inds = sims_proj.sort(dim=0, descending=True)  # (V,)
            src_sims[index] = 1.0  # NOTE: set the target view similarity to 1.0

            # Compute the distance similarity
            inds_dist = src_inds[src_sims > args.threshold_proj]  # (Vd,)
            sims_dist = compute_relative_dists(tar_ext[None], src_exts[inds_dist])[0]  # (Vd,)
            src_sims_dist, src_inds_dist = sims_dist.sort(dim=0, descending=False)  # (Vd,)
            # Update the source similarity and index
            src_inds[:len(src_inds_dist)] = inds_dist[src_inds_dist]

            # Compute the rotation similarity
            inds_rots = src_inds[src_sims > args.threshold_proj][src_sims_dist < args.threshold_dist]  # (Vr)
            sims_rots = compute_relative_angle(tar_ext[None], src_exts[inds_rots])[0]  # (Vr,)
            src_sims_rots, src_inds_rots = sims_rots.sort(dim=0, descending=False)  # (Vr,)
            # Update the source similarity and index
            src_inds[:len(src_inds_rots)] = inds_rots[src_inds_rots]

            # Save the source view similarity and index
            scene_sims[index] = src_sims
            scene_inds[index] = src_inds

        # Generate the iterable of views
        indices = list(range(len(camera_names)))
        camears = camera_names.tolist()
        # Process each subscene parallel
        parallel_execution(indices, camears, action=process_view, sequential=True, print_progress=True)

        # Save the source similarity and index
        np.save(join(output_dir, 'src_sims.npy'), scene_sims.numpy())
        np.save(join(output_dir, 'src_inds.npy'), scene_inds.numpy())

    # Use the cached scenes if exists
    if exists(join(data_root, 'scenes.json')):
        scenes = json.load(open(join(data_root, 'scenes.json')))['scenes']
    else:
        # Find and process all scenes
        scenes = sorted(glob(join(data_root, "**", "extri.yml"), recursive=True))
        scenes = [dirname(p) for p in scenes]
        # Cache for acceleration
        json.dump(dict(scenes=scenes), open(join(data_root, 'scenes.json'), 'w'), indent=4)

    if args.scene is not None:
        log(yellow(f'Only process scene: {args.scene}'))
        scenes = [x for x in scenes if args.scene in x]
    parallel_execution(scenes, action=process_scene, sequential=True, print_progress=True)  # literally a for-loop


if __name__ == '__main__':
    main()

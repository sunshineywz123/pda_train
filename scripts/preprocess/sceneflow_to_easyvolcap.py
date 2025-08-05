# This converts the SceneFlow dataset to easyvolcap format: link images directly
# https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html

import os
import argparse
import numpy as np
from glob import glob
from os.path import join, dirname
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.data_utils import save_image
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.easy_utils import read_camera, write_camera


def load_pfm(file_path):
    # Open the file and read the header
    with open(file_path, encoding="ISO-8859-1") as fp:
        color = None
        width = None
        height = None
        scale = None
        endian = None

        # Load file header and grab channels, if is 'PF' 3 channels else 1 channel(gray scale)
        header = fp.readline().rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        # Grab image dimensions
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', fp.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        # Grab image scale
        scale = float(fp.readline().rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        # Grab image data
        data = np.fromfile(fp, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        # Reshape data to [Height, Width, Channels]
        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data, scale


def load_camera_data(filename):
    # Load the original camera data
    with open(filename, 'r') as file:
        lines = file.readlines()

    lc2w, rc2w = [], []
    # Parse the original camera data
    for line in lines:
        elements = line.strip().split()
        # Skip the line if it is not a camera matrix
        if len(elements) == 0 or (elements[0] != 'L' and elements[0] != 'R'): continue

        # Parse the camera matrix
        c2w = np.array(list(map(float, elements[1:])), dtype=float).reshape(4, 4)
        if elements[0] == 'L': lc2w.append(c2w)
        if elements[0] == 'R': rc2w.append(c2w)

    # Convert the camera matrices to np.ndarray
    lc2w, rc2w = np.array(lc2w, dtype=float), np.array(rc2w, dtype=float)  # (N, 4, 4)
    c2ws = np.stack([lc2w, rc2w], axis=0)  # (2, N, 4, 4)

    # Convert from Blender/OpenGL coordinate system to OpenCV coordinate system
    c2ws[..., 1:3] = -c2ws[..., 1:3]

    return c2ws


# Define some global variables for SceneFlow
sceneflow_cam_dir = 'camera_data'
sceneflow_img_dir = 'frames_cleanpass'
sceneflow_dsp_dir = 'disparity'
K0 = np.array([[1050.0, 0.0, 479.5], [0.0, 1050.0, 269.5], [0.0, 0.0, 1.0]], dtype=np.float32)
K1 = np.array([[ 450.0, 0.0, 479.5], [0.0,  450.0, 269.5], [0.0, 0.0, 1.0]], dtype=np.float32)

# Define some global variables for EasyVolcap
easyvolcap_cam_dir = 'cameras'
easyvolcap_img_dir = 'images'
easyvolcap_dsp_dir = 'disparities'


@catch_throw
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sceneflow_root', type=str, default='data/workspace/SceneFlow')
    parser.add_argument('--easyvolcap_root', type=str, default='data/workspace/scene_flow')
    parser.add_argument('--scene', type=str, default=None)
    args = parser.parse_args()

    sceneflow_root = args.sceneflow_root
    easyvolcap_root = args.easyvolcap_root
    H, W = 540, 960

    def process_scene(scene: str):
        # Find all subdirectories under the scene directory
        subscenes = [dirname(p).replace(join(sceneflow_root, scene, f'{sceneflow_cam_dir}/'), '') for p in glob(join(sceneflow_root, scene, '**', 'camera_data.txt'), recursive=True)]
        subscenes = sorted(subscenes)

        def process_subscene(subscene: str):
            # The input easyvolcap format data directory is for sure
            subscene_in_cam_dir = join(sceneflow_root, scene, sceneflow_cam_dir, subscene)
            subscene_in_img_dir = join(sceneflow_root, scene, sceneflow_img_dir, subscene)
            subscene_in_dsp_dir = join(sceneflow_root, scene, sceneflow_dsp_dir, subscene)

            # The output easyvolcap format data directory is for sure
            subscene_out_cam_dir = join(easyvolcap_root, scene, subscene, easyvolcap_cam_dir)
            subscene_out_img_dir = join(easyvolcap_root, scene, subscene, easyvolcap_img_dir)
            subscene_out_dsp_dir = join(easyvolcap_root, scene, subscene, easyvolcap_dsp_dir)

            # Read and parse the raw camera data
            c2ws = load_camera_data(join(subscene_in_cam_dir, 'camera_data.txt'))  # (2, N, 4, 4)

            # Process the left view and right view respectively
            for v, view in enumerate(['left', 'right']):
                # One camera for each view
                cameras = dotdict()

                # Sorted image names and disparity names
                ims = sorted(os.listdir(join(subscene_in_img_dir, view)))
                dps = sorted(os.listdir(join(subscene_in_dsp_dir, view)))

                # Create the output directories
                os.makedirs(join(subscene_out_cam_dir, f'{v}'), exist_ok=True)
                os.makedirs(join(subscene_out_img_dir, f'{v}'), exist_ok=True)
                os.makedirs(join(subscene_out_dsp_dir, f'{v}'), exist_ok=True)

                for f in range(c2ws.shape[1]):
                    # Camera data
                    cameras[f'{f:06d}'] = dotdict()  # might have more than 100 cameras?
                    cam = cameras[f'{f:06d}']
                    cam.H = H
                    cam.W = W
                    cam.K = K0 if '15mm' not in subscene else K1
                    cam.R = c2ws[v, f, :3, :3].T  # 3, 3
                    cam.T = -cam.R @ c2ws[v, f, :3, 3]  # 3, 3 @ 3
                    cam.D = np.zeros((1, 5))

                    # Link the image
                    os.system(f"ln -s {os.path.abspath(join(subscene_in_img_dir, view, ims[f]))} {join(subscene_out_img_dir, f'{v}', f'{f:06d}.jpg')}")

                    # Load and save the disparity to `.exr`
                    dsp, scale = load_pfm(join(subscene_in_dsp_dir, view, dps[f]))
                    save_image(join(subscene_out_dsp_dir, f'{v}', f"{f:06d}.exr"), dsp)

                # Write the camera data
                write_camera(cameras, join(subscene_out_cam_dir, f'{v}'))  # extri.yml and intri.yml

        # Process each subscene parallel
        parallel_execution(subscenes, action=process_subscene, sequential=False, print_progress=True)  # literally a for-loop

    # Find all the three scenes under the SceneFlow root
    scenes = sorted(os.listdir(sceneflow_root))
    if args.scene is not None:
        log(yellow(f'Only process scene: {args.scene}'))
        scenes = [x for x in scenes if args.scene in x]
    parallel_execution(scenes, action=process_scene, sequential=True, print_progress=True)  # literally a for-loop


if __name__ == '__main__':
    main()

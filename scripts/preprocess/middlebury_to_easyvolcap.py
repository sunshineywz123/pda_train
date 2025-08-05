# This converts the middlebury dataset to easyvolcap format: link images directly

# Note that the original middlebury dataset does not contain the camera extrinsic parameters, and
# what I did here is to assume the left view is the identity matrix and the right view is -baseline
# away from the left view.

# The converted results have been verified via depth fusion.

# https://vision.middlebury.edu/stereo/data/scenes2014/
# https://vision.middlebury.edu/stereo/eval3/MiddEval3-newFeatures.html

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


def parse_calib(filepath: str) -> dict:
    # Open the raw `calib.txt`
    with open(filepath, 'r') as file: lines = file.readlines()

    res = {}
    # Parsing the values
    for line in lines:
        key, value = line.split("=")
        # If the value contains ; , it must be a np.ndarray
        if ";" in value:
            mat = []
            value = value.strip()[1:-1]
            rows = value.split(";")
            for row in rows:
                if row.strip() != '':
                    mat.append([float(x) for x in row.split()])
            res[key] = np.array(mat)
        elif "." in value:  # If the value contains . , it must be a float
            res[key] = float(value)
        else:  # Otherwise, it is int
            res[key] = int(value)
            
    return res


# Define some global variables for EasyVolcap
easyvolcap_img_dir = 'images'
easyvolcap_dpt_dir = 'depths'
easyvolcap_dsp_dir = 'disparities'


@catch_throw
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--middlebury_root', type=str, default='/mnt/data/Datasets/middlebury/MiddEval3')
    parser.add_argument('--easyvolcap_root', type=str, default='data/workspace/middlebury')
    args = parser.parse_args()

    middlebury_root = args.middlebury_root
    easyvolcap_root = args.easyvolcap_root

    # Record the min and max depth
    range_dict = dict(scenes=dict())
    range_list = []

    def process_scene(scene: str, split: str):
        s = 'train' if 'train' in split else 'test'
        scene_root = join(easyvolcap_root, s, scene)

        # Parse the `calib.txt`
        calib = parse_calib(join(middlebury_root, split, scene, 'calib.txt'))
        H, W, f = calib['height'], calib['width'], calib['cam0'][0, 0]

        # Placeholder for both left and right camera
        cameras = dotdict()

        # Deal with the left view and right view respectively
        for i in range(2):
            # Generate easyvolcap camera first
            cameras[f'{i}'] = dotdict()
            cam = cameras[f'{i}']
            cam.H, cam.W = H, W
            cam.K = calib[f'cam{i}']
            cam.R = np.eye(3)  # (3, 3)
            cam.T = np.zeros((3)) if i == 0 else np.zeros((3)) - np.array([calib['baseline'] / 1000., 0, 0])  # (3)
            cam.D = np.zeros((1, 5))  # (1, 5)

            # Create the output directory for rgb image
            os.makedirs(join(scene_root, easyvolcap_img_dir, f'{i}'), exist_ok=True)
            # Link the image
            os.system(f"ln -s {os.path.abspath(join(middlebury_root, split, scene, f'im{i}.png'))} {join(scene_root, easyvolcap_img_dir, f'{i}', f'{0:06d}.png')}")

            if s != 'test':
                # Create the output directory for disparity and depth map
                os.makedirs(join(scene_root, easyvolcap_dpt_dir, f'{i}'), exist_ok=True)
                os.makedirs(join(scene_root, easyvolcap_dsp_dir, f'{i}'), exist_ok=True)
                # Read in the disparity map
                dsp, scale = load_pfm(join(middlebury_root, split, scene, f'disp{i}GT.pfm'))
                dpt = calib['baseline'] * f / (dsp + calib['doffs']) / 1000.
                # Write the disparity map and depth map
                save_image(join(scene_root, easyvolcap_dsp_dir, f'{i}', f'{0:06d}.exr'), dsp)
                save_image(join(scene_root, easyvolcap_dpt_dir, f'{i}', f'{0:06d}.exr'), dpt)
                # Compute the min and max depth
                min_dpt, max_dpt = np.percentile(dpt, 2.), np.percentile(dpt, 98.)
                range_dict['scenes'][f'{scene}'] = (min_dpt, max_dpt)
                range_list.append((min_dpt, max_dpt))

        # Write the camera data
        write_camera(cameras, scene_root)  # extri.yml and intri.yml

    # Find all the three scenes under the middlebury root
    for split in ['trainingQ', 'testQ']:
        scenes = sorted(os.listdir(join(middlebury_root, split)))
        splits = [split for _ in range(len(scenes))]
        parallel_execution(scenes, splits, action=process_scene, sequential=True, print_progress=True)  # literally a for-loop

    # Compute the average min and max depth
    range_list = np.array(range_list)
    min_dpt, max_dpt = np.mean(range_list[:, 0]), np.mean(range_list[:, 1])
    range_dict['average'] = (min_dpt, max_dpt)
    # Save the range dict
    json.dump(range_dict, open(join(easyvolcap_root, 'train', 'depth_ranges.json'), 'w'), indent=4)
    print(f"Average min depth: {min_dpt:.3f}, max depth: {max_dpt:.3f}")


if __name__ == '__main__':
    main()

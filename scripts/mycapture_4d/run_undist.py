import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm

import cv2
sys.path.append('.')
from trdparties.colmap.read_write_model import read_model
from lib.utils.undist_utils import colmap_undistort_numpy
from lib.utils.parallel_utils import async_call

def main(args):
    input = args.input
    colmap_path = join(input, 'colmap/sparse/0')
    cameras, images, points3D = read_model(colmap_path)
    name2imid = {images[imid].name: imid for imid in images}

    @async_call
    def undist(input_path, output_path, K, D, quality=95):
        img = cv2.imread(input_path)
        dst, Kt = colmap_undistort_numpy(img, K, D)
        if '.png' in output_path: cv2.imwrite(output_path, dst)
        else: cv2.imwrite(output_path, dst, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

    videos = sorted(os.listdir(join(input, 'images')))
    for video in tqdm(videos):
        camera = cameras[images[name2imid[video + '.jpg']].camera_id]
        K = np.eye(3)
        D = np.zeros(5)
        K[0, 0], K[1, 1] = camera.params[0], camera.params[1] 
        K[0, 2], K[1, 2] = camera.params[2], camera.params[3]
        D[:4] = camera.params[4:8]
        for frame_id in tqdm(sorted(os.listdir(join(input, 'images', videos[0])))):
            img_path = join(input, 'images', video, frame_id)
            output_img_path = join(input, 'images_undist', video, frame_id)
            os.makedirs(join(input, 'images_undist', video), exist_ok=True)
            undist(img_path, output_img_path, K, D, quality=95)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/bn/haotongdata/Datasets/mycapture_stray_scanner/20240728_seq2')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)
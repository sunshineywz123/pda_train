import os
from os.path import join
import argparse
import sys
import numpy as np
import cv2
from tqdm import tqdm
import shutil

def variance_of_laplacian(image):
    """Compute the variance of the Laplacian of the image."""
    return cv2.Laplacian(image, cv2.CV_64F).var()

def detect_blurry_frames(input_dir, output_clear_dir, output_blurry_dir, threshold_factor=0.01):
    os.system('rm -rf ' + output_clear_dir)
    os.system('rm -rf ' + output_blurry_dir)
    if not os.path.exists(output_clear_dir):
        os.makedirs(output_clear_dir)
    if not os.path.exists(output_blurry_dir):
        os.makedirs(output_blurry_dir)

    sharpness_values = []
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Calculate sharpness for each image
    for image_file in tqdm(image_files, desc="Calculating sharpness"):
        image_path = join(input_dir, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            sharpness = variance_of_laplacian(image)
            sharpness_values.append((image_file, sharpness))
    sharpness_values.sort(key=lambda x: x[1])
    sharpness_only = [x[1] for x in sharpness_values]
    mean_sharpness = np.mean(sharpness_only)
    std_sharpness = np.std(sharpness_only)
    threshold = mean_sharpness - threshold_factor * std_sharpness

    # Move images to corresponding directories based on sharpness
    for image_file, sharpness in tqdm(sharpness_values, desc="Sorting images"):
        src_path = join(input_dir, image_file)
        if sharpness < threshold:
            dst_path = join(output_blurry_dir, image_file)
        else:
            dst_path = join(output_clear_dir, image_file)
        shutil.copy(src_path, dst_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_rgb_dir', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/98b4ec142f/iphone/colmap_sfm/images')
    parser.add_argument('--output_clear_rgb_dir', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/98b4ec142f/merge/images/iphone')
    parser.add_argument('--output_blurry_rgb_dir', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/98b4ec142f/merge/trash')
    args = parser.parse_args()
    return args

def main(args):
    detect_blurry_frames(args.input_rgb_dir, args.output_clear_rgb_dir, args.output_blurry_rgb_dir)

if __name__ == '__main__':
    args = parse_args()
    main(args)
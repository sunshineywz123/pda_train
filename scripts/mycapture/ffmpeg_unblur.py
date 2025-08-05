import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
from scripts.mycapture.run import detect_blurry_frames

def main(args):
    os.makedirs(args.output, exist_ok=True)
    temp_dir = args.input + '_temp1'
    os.makedirs(temp_dir)
    os.system('ffmpeg -i {} -start_number 0  -q:v 2 {}/%06d.jpg'.format(args.input, temp_dir))
    detect_blurry_frames(
        temp_dir, 
        args.output, 
        top_percent=24, 
        max_interval=60, 
        min_interval=6,
        soft_link=False,
        max_number=500)
    os.system('rm -rf {}'.format(temp_dir))
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/bn/haotongdata/Datasets/mycapture_stray_scanner/bbc40c11e0/rgb.mp4')
    parser.add_argument('--output', type=str, default='/mnt/bn/haotongdata/Datasets/mycapture_stray_scanner/bbc40c11e0/images/arkit')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)

import os
from os.path import join
import argparse
import shutil
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')

def main(args):
    image_dir = join(args.output, 'images')
    depth_dir = join(args.output, 'depth')
    conf_dir = join(args.output, 'confidence')
    for frame in tqdm(os.listdir(join(args.input, 'bkgd_scan'))):
        src = join(args.input, 'bkgd_scan', frame)
        dst = join(image_dir, 'bkgd_scan', 'bkgd_frame_' + frame)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
        src = join(args.bkgd_scan_depth, frame.replace('.jpg', '.png'))
        dst = join(depth_dir, 'bkgd_scan', 'bkgd_frame_' + frame.replace('.jpg', '.png'))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

        src = join(args.bkgd_scan_conf, frame.replace('.jpg', '.png'))
        dst = join(conf_dir, 'bkgd_scan', 'bkgd_frame_' + frame.replace('.jpg', '.png'))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

    for video in os.listdir(args.input + '/images'):
        src = join(args.input, 'bkgd', video + '.jpg')
        dst = join(image_dir, f'bkgd_{video}', 'bkgd_' + video + '.jpg')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
        src = join(args.input, 'bkgd_depth', video + '.png')
        dst = join(depth_dir, f'bkgd_{video}', 'bkgd_' + video + '.png')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
        src = join(args.input, 'bkgd_conf', video + '.png')
        dst = join(conf_dir, f'bkgd_{video}', 'bkgd_' + video + '.png')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/data/home/linhaotong/datasets/mycapture4d/20240728_seq2')
    parser.add_argument('--bkgd_scan_depth', type=str, default='/mnt/data/home/linhaotong/datasets/mycapture4d/20240728/31/background_scan/depth')
    parser.add_argument('--bkgd_scan_conf', type=str, default='/mnt/data/home/linhaotong/datasets/mycapture4d/20240728/31/background_scan/confidence')
    parser.add_argument('--output', type=str, default='/mnt/data/home/linhaotong/datasets/mycapture4d/20240728_seq2/calib')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)
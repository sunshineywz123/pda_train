import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')

def main(args):
    tag = -1
    input = args.input
    video = args.video

    # image
    images = sorted(os.listdir(join(input, 'images', video)))
    if tag < 0:
        for image in images:
            if int(image.split(".")[0]) + tag < 0: continue

            src = join(input, 'images', video, image)
            dst = join(input, 'images', video, f'{int(image.split(".")[0]) + tag :06d}.jpg')
            os.rename(src, dst)

            src = join(input, 'depth', video, image.replace('.jpg', '.png'))
            dst = join(input, 'depth', video, f'{int(image.split(".")[0]) + tag :06d}.png')
            os.rename(src, dst)

            # src = join(input, 'images_undist', video, image)
            # dst = join(input, 'images_undist', video, f'{int(image.split(".")[0]) + tag :06d}.jpg')
            # os.rename(src, dst)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/bn/haotongdata/Datasets/mycapture_stray_scanner/20240728_seq1')
    parser.add_argument('--video', type=str, default='31')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)
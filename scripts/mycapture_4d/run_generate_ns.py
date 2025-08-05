import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm

from lib.utils.pylogger import Log
sys.path.append('.')

def main(args):
    input = args.input
    frame = args.frame
    output_dir = join(input, 'ns_output/{:06d}/images'.format(frame))
    os.makedirs(output_dir, exist_ok=True)
    for cam in os.listdir(join(input, 'images')):
        src_path = join(input, 'images', cam, '{:06d}.jpg'.format(frame))
        tar_path = join(output_dir, '{}.jpg'.format(cam))
        os.system('cp {} {}'.format(src_path, tar_path))
    Log.info('write to {}'.format(output_dir))
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/bn/haotongdata/Datasets/mycapture_stray_scanner/seq2_raw/seq2_mocap')
    parser.add_argument('--frame', type=int, default=0)
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)
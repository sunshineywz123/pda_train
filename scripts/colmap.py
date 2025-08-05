import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
from lib.utils.pylogger import Log

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/mnt/bn/haotongdata/Datasets/tat/test_scenes/Meetingroom')
    parser.add_argument('--matcher', type=str, default='sequential_matcher', help='exhaustive_matcher or sequential_matcher')
    parser.add_argument('--extract_gpu', action='store_true')
    parser.add_argument('--do_all', action='store_true')
    args = parser.parse_args()
    return args

def main(args):
    root_dir = args.input_dir
    output_dir = join(args.input_dir, 'colmap')
    if args.extract_gpu: os.system('rm -rf ' + output_dir)
    Log.info(f'Output dir: {output_dir}')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(join(output_dir, 'sparse'), exist_ok=True)
    
    cmd = f'colmap feature_extractor --database_path {output_dir}/database.db --image_path {root_dir}/images --ImageReader.single_camera 1 --ImageReader.camera_model PINHOLE'
    Log.info('Feature extraction started')
    Log.info(cmd)
    if args.do_all or args.extract_gpu:
        os.system(cmd)
    Log.info('Feature extraction done')
    
    cmd = f'colmap {args.matcher} --database_path {output_dir}/database.db'
    Log.info('matching started')
    Log.info(cmd)
    if args.do_all or args.extract_gpu:
        os.system(cmd)
    Log.info('matching done')
    
    Log.info('Mapper started')
    cmd = f'colmap mapper --database_path {output_dir}/database.db --image_path {root_dir}/images --output_path {output_dir}/sparse'
    Log.info(cmd)
    if args.do_all or not args.extract_gpu:
        os.system(cmd)
    Log.info('Mapper done')

if __name__ == '__main__':
    args = parse_args()
    main(args)
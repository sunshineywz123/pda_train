import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
import glob
import subprocess
from lib.utils.pylogger import Log

from trdparties.colmap.read_write_model import read_images_binary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='/mnt/remote/D002/home/linhaotong/3d_scanner')
    parser.add_argument('--input_tag', type=str, default='orig')
    parser.add_argument('--output_tag', type=str, default='processed')
    parser.add_argument('--scene', type=str, default='2024_04_16_16_13_19')
    
    args = parser.parse_args()
    return args

def main(args):
    input_dir = join(args.base_dir, args.input_tag, args.scene)
    jpgs = glob.glob(join(input_dir, 'frame_*.jpg'))
    
    # copy to base_dir, input_tag, scene, 'rgb'
    target_dir = join(args.base_dir, args.output_tag, args.scene, 'rgb')
    os.makedirs(target_dir, exist_ok=True)
    
    # copy images
    for jpg in tqdm(jpgs):
        os.system(f'cp {jpg} {target_dir}')
        
    
    # colmap
    # colmap feature_extractor --database database.db --image_path rgb --ImageReader.single_camera 1 --ImageReader.camera_model=PINHOLE 
    # colmap exhaustive_matcher --database_path database.db
    # mkdir -p sparse
    # colmap mapper --database_path database.db --image_path rgb --output_path sparse 
    
    basedir = join(args.base_dir, args.output_tag, args.scene)
    os.makedirs(join(basedir, 'colmap'), exist_ok=True)
    command_args = [
        'colmap', 'feature_extractor', 
            '--database_path', os.path.join(basedir, 'colmap/database.db'), 
            '--image_path', os.path.join(basedir, 'rgb'),
            '--ImageReader.single_camera', '1',
            '--ImageReader.camera_model', 'PINHOLE',
    ]
    subprocess.check_output(command_args, universal_newlines=True)
    Log.info('feature_extractor done')
    
    command_args = [
        'colmap', 'exhaustive_matcher', 
            '--database_path', os.path.join(basedir, 'colmap/database.db'), 
    ]
    subprocess.check_output(command_args, universal_newlines=True)
    Log.info('exhaustive_matcher done')
    
    os.makedirs(join(basedir, 'colmap/sparse'), exist_ok=True)
    command_args = [
        'colmap', 'mapper',
            '--database_path', os.path.join(basedir, 'colmap/database.db'),
            '--image_path', os.path.join(basedir, 'rgb'),
            '--output_path', os.path.join(basedir, 'colmap/sparse')
    ]
    subprocess.check_output(command_args, universal_newlines=True)
    Log.info('mapper done')
    
    registered_imgs_len = len(read_images_binary(join(basedir, 'colmap/sparse/0/images.bin')))
    overall_imgs_len = len(jpgs)
    Log.info(f'registered_imgs_len: {registered_imgs_len}, overall_imgs_len: {overall_imgs_len}')
    
    command_args = [
        'colmap', 'model_orientation_aligner',
            '--input_path', os.path.join(basedir, 'colmap/sparse/0'),
            '--output_path', os.path.join(basedir, 'colmap/sparse'),
            '--image_path', os.path.join(basedir, 'rgb'),
    ]
    subprocess.check_output(command_args, universal_newlines=True)
    Log.info('model_aligner done')
    
    os.system('rm -rf {}'.format(os.path.join(basedir, 'colmap/sparse/0')))
    Log.info('remove sparse/0 done')
    
    
    os.makedirs(join(basedir, 'conf'), exist_ok=True)
    os.makedirs(join(basedir, 'depth'), exist_ok=True)
    images = read_images_binary(join(basedir, 'colmap/sparse/images.bin'))
    for k in images:
        image = images[k]
        src_path = join(args.base_dir, args.input_tag, args.scene, image.name.replace('frame_', 'conf_')[:-4] + '.png')
        tar_path = join(basedir, 'conf', image.name.replace('frame_', 'conf_')[:-4] + '.png')
        os.system('cp {} {}'.format(src_path, tar_path))
        
        src_path = join(args.base_dir, args.input_tag, args.scene, image.name.replace('frame_', 'depth_')[:-4] + '.png')
        tar_path = join(basedir, 'depth', image.name.replace('frame_', 'depth_')[:-4] + '.png')
        os.system('cp {} {}'.format(src_path, tar_path))
    Log.info('copy conf and depth done')
        
if __name__ == '__main__':
    args = parse_args()
    main(args)
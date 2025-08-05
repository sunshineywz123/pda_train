import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/data/Datasets/DL3DV')
    args = parser.parse_args()
    return args

def main(args):
    deletes = ['images', 'images_2', 'images_4', 'images_8']
    
    for file in os.scandir(args.input):
        if file.is_dir():
            delete_path = join(file.path, 'gaussian_splat/input')
            if os.path.exists(delete_path):
                os.system(f'rm -rf {delete_path}')
            for delete in deletes:
                delete_path = join(file.path, 'gaussian_splat', delete)
                orig_path = join('../nerfstudio', delete)
                if os.path.exists(delete_path):
                    os.system(f'rm -rf {delete_path}')
                os.system(f'ln -s {orig_path} {delete_path}')
            # remove input, images in gaussian
            # then create soft link
        

if __name__ == '__main__':
    args = parse_args()
    main(args)
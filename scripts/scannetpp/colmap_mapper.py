import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
from lib.utils.pylogger import Log
from trdparties.colmap.read_write_model import read_images_binary


def get_model_num(path):
    images = read_images_binary(join(path, 'images.bin'))
    return len(images)

def report_model(path):
    num = get_model_num(path)
    Log.info(f'{path} has {num} images')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/56a0ec536c/merge_dslr_iphone')
    parser.add_argument('--scene', type=str, default=None)
    args = parser.parse_args()
    return args

def main(args):
    root_dir = args.input_dir
    if args.scene is not None: root_dir = root_dir.replace('56a0ec536c', args.scene)
    scene = '56a0ec536c' if args.scene is None else args.scene
    output_dir = join(root_dir, 'colmap')
    sparse_dir = join(root_dir, 'colmap/sparse')
    os.system('rm -rf ' + sparse_dir)
    os.makedirs(sparse_dir, exist_ok=True)
    cmd = f'colmap mapper --database_path {output_dir}/database.db --image_path {root_dir}/images --output_path {sparse_dir} --log_to_stderr 0'
    os.system(cmd)
    models = os.listdir(sparse_dir)
    if len(models) != 0: models = [model for model in models if '.ini' not in model]
    if len(models) == 0: num = 0
    elif len(models) == 1: num = get_model_num(join(sparse_dir, models[0]))
    elif len(models) > 1:
        nums = []
        for model in models:
            nums.append(get_model_num(join(sparse_dir, model)))
        max_num = max(nums)
        max_model = models[nums.index(max_num)]
        if max_model != '0':
            os.system(f'mv {join(sparse_dir, "0")} {join(sparse_dir, str(len(models)))}')
            os.system(f'mv {join(sparse_dir, max_model)} {join(sparse_dir, "0")}')
        num = max_num
    open('mapper.txt', 'a').write(scene + ': {}\n'.format(num))

if __name__ == '__main__':
    args = parse_args()
    main(args)
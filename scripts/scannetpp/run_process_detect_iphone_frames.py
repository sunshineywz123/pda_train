import json
import os
from os.path import join
import argparse
import sys
import ipdb
import numpy as np
from tqdm import tqdm

from lib.utils.pylogger import Log
from lib.utils.parallel_utils import parallel_execution
sys.path.append('.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp')
    args = parser.parse_args()
    return args

def run_cmd(cmd):
    os.system(cmd)

def main(args):
    # 统计每个scene下面DSLR的图片有多少
    # 统计每个scene下面iphone的图片有多少
    # 统计每个scene下面depth的图片有多少
    # 检查每个scene下面iphone和depth图像是不是一一对应
    # run_statics(args)
    # run_plots(args)
    train_scenes = open(join(args.input, 'splits/nvs_sem_train.txt')).readlines()
    val_scenes = open(join(args.input, 'splits/nvs_sem_val.txt')).readlines()
    scenes = train_scenes + val_scenes
    scenes = [scene.strip() for scene in scenes]
    scenes = sorted(scenes)
    cmds = ['python3 scripts/scannetpp/run_detect_blurry_complex.py  --scene {}'.format(scene) for scene in scenes]
    parallel_execution(
        cmds,
        action=run_cmd,
        num_processes=16,
        print_progress=True
    )

if __name__ == '__main__':
    args = parse_args()
    main(args)
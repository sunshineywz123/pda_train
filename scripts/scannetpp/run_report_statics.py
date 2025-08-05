import json
import os
from os.path import join
import argparse
import sys
import ipdb
import numpy as np
from tqdm import tqdm

from lib.utils.pylogger import Log
sys.path.append('.')




def run_statics(args):
    info = {}
    train_scenes = open(join(args.input, 'splits/nvs_sem_train.txt')).readlines()
    val_scenes = open(join(args.input, 'splits/nvs_sem_val.txt')).readlines()
    scenes = train_scenes + val_scenes
    scenes = [scene.strip() for scene in scenes]
    scenes = sorted(scenes)
    
    for scene in tqdm(scenes):
        image_num = os.listdir(join(args.input, 'data', scene, 'iphone/rgb'))
        depth_num = os.listdir(join(args.input, 'data', scene, 'iphone/depth'))
        dslr_num = os.listdir(join(args.input, 'data', scene, 'dslr/resized_images'))
        if len(depth_num) != len(image_num) and  (len(depth_num)+1) != len(image_num):
            import ipdb; ipdb.set_trace()
        info[scene] = {
            'image_num': len(image_num),
            'dslr_num': len(dslr_num)
        }
    with open(args.output_json, 'w') as f:
        json.dump(info, f)
        Log.info(f'Save statics info to {args.output_json}')
        
def run_plots(args):
    info = json.load(open(args.output_json, 'r'))
    import matplotlib.pyplot as plt 
    scenes = list(info.keys())
    image_nums = [info[scene]['image_num'] for scene in scenes]
    dslr_nums = [info[scene]['dslr_num'] for scene in scenes]
    import ipdb; ipdb.set_trace()
    image_nums = sorted(image_nums)
    dslr_nums = sorted(dslr_nums)
    # np.median(image_nums), 7938
    # np.max(image_nums), 35475
    # np.median(dslr_nums), 423, 2704
    # 保证iphone images, 平均1/30, 平均300, 最多1000
    # 如果不超过600, 就保留全部
    # 如果大于600, 就随机选600张
    plt.plot(np.arange(len(scenes)), image_nums, '.', label='image_nums')
    plt.plot(np.arange(len(scenes)), dslr_nums, '.', label='dslr_nums')
    plt.savefig(join(args.output_dir, args.output_name))
    Log.info(f'Save plot to {join(args.output_dir, args.output_name)}')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp')
    parser.add_argument('--output_json', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/info.json')
    parser.add_argument('--output_dir', type=str, default='/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/jpgs')
    parser.add_argument('--output_name', type=str, default='scannetpp_stastics.jpg')
    args = parser.parse_args()
    return args

def main(args):
    # 统计每个scene下面DSLR的图片有多少
    # 统计每个scene下面iphone的图片有多少
    # 统计每个scene下面depth的图片有多少
    # 检查每个scene下面iphone和depth图像是不是一一对应
    # run_statics(args)
    run_plots(args)

if __name__ == '__main__':
    args = parse_args()
    main(args)
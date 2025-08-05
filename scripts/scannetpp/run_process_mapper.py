import json
import os
from os.path import join
import argparse
import sys
import ipdb
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from lib.utils.pylogger import Log
from lib.utils.parallel_utils import parallel_execution
import time
sys.path.append('.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp')
    parser.add_argument('--batch_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_processes', type=int, default=10)
    parser.add_argument('--sleep_time', type=int, default=600)
    args = parser.parse_args()
    return args

def run_cmd(cmd):
    os.system(cmd)
    
def get_num_images(scene):
    scannetpp_dir = f'/mnt/bn/haotongdata/Datasets/scannetpp/data/{scene}/merge_dslr_iphone/images'
    num = len(os.listdir(join(scannetpp_dir, 'iphone'))) + len(os.listdir(join(scannetpp_dir, 'dslr')))
    return num

def get_num_models(scene):
    scannetpp_dir = f'/mnt/bn/haotongdata/Datasets/scannetpp/data/{scene}/merge_dslr_iphone/colmap/sparse'
    if not os.path.exists(scannetpp_dir): return 0
    models = os.listdir(scannetpp_dir)
    if len(models) != 0: models = [model for model in models if '.ini' not in model]
    return len(models)

def main(args):
    train_scenes = open(join(args.input, 'splits/nvs_sem_train.txt')).readlines()
    val_scenes = open(join(args.input, 'splits/nvs_sem_val.txt')).readlines()
    scenes = train_scenes + val_scenes
    scenes = [scene.strip() for scene in scenes]
    scenes = sorted(scenes)
    scenes = scenes[args.batch_id * args.batch_size: (args.batch_id + 1) * args.batch_size]
    if args.batch_id * args.batch_size <= 80:
        model_nums = [get_num_models(scene) for scene in scenes]
        scenes = [scene for model_num, scene in zip(model_nums, scenes) if model_num == 0]
    # ["3c95c89d61", {"num_iphone": 317, "num_dslr": 400, "num_regi_iphone": 12, "num_regi_dslr": 0, "regi_ratio": 0.016736401673640166}]
    # ["394a542a19", {"num_iphone": 243, "num_dslr": 620, "num_regi_iphone": 0, "num_regi_dslr": 3, "regi_ratio": 0.0034762456546929316}]
    # ["355e5e32db", {"num_iphone": 555, "num_dslr": 1000, "num_regi_iphone": 0, "num_regi_dslr": 2, "regi_ratio": 0.0012861736334405145}]
    scenes = ['3c95c89d61', '394a542a19', '355e5e32db']
    nums = [get_num_images(scene) for scene in scenes]
    sorted_scenes = [scene for _, scene in sorted(zip(nums, scenes), reverse=True)]
    cmds = ['python3 scripts/scannetpp/colmap_mapper.py --scene {}'.format(scene) for scene in sorted_scenes]
    print(f"Processing scenes: {sorted_scenes}")
    results = []
    with Pool(args.num_processes) as pool:
        for i, cmd in enumerate(cmds):
            result = pool.apply_async(run_cmd, (cmd,))
            results.append(result)
            print(f"Executing command: {cmd}")
            time.sleep(args.sleep_time)  # 延迟15分钟
        for result in results:
            result.get()
    pool.close()
    pool.join()
    print("All commands have been executed.")
    

if __name__ == '__main__':
    args = parse_args()
    main(args)
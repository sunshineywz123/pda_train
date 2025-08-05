import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm

from lib.utils.pylogger import Log
import json
sys.path.append('.')

def main(args):
    data_root = args.input
    nvs_sem_val_scenes = [scene.strip() for scene in open(join(data_root, 'nvs_sem_val.txt')).readlines()]
    nvs_sem_train_scenes = [scene.strip() for scene in open(join(data_root, 'nvs_sem_train.txt')).readlines()]
    train_scenes = [scene.strip() for scene in open(join(data_root, 'train_scenes_0614.txt')).readlines()]
    test_scenes = [scene.strip() for scene in open(join(data_root, 'test_scenes_0614.txt')).readlines()]

    remain_val_scenes = []
    for scene in nvs_sem_val_scenes:
        if scene not in train_scenes and scene not in test_scenes:
            remain_val_scenes.append(scene)
    
    remain_train_scenes = []
    for scene in nvs_sem_train_scenes:
        if scene not in train_scenes and scene not in test_scenes:
            remain_train_scenes.append(scene)

    Log.info(f'Remain train scenes: {len(remain_train_scenes)}')
    Log.info(remain_train_scenes)

    Log.info(f'Remain val scenes: {len(remain_val_scenes)}')
    Log.info(remain_val_scenes)

    regi_ratio = {}
    regi_ratio_path = '/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/txts/scannetpp/regi_ratio_scene_infos.json'
    for line in open(regi_ratio_path).readlines():
        scene_info = json.loads(line)
        regi_ratio[scene_info[0]] = scene_info[1]
        scene = scene_info[0]

    # /mnt/bn/haotongdata/Datasets/scannetpp/data/ac48a9b736/scans
    for scene in remain_train_scenes:
        Log.info(f'{scene}: {regi_ratio[scene]}')
        Log.info(f'/mnt/bn/haotongdata/Datasets/scannetpp/data/{scene}/scans/mesh_aligned_0.05.ply')

    for scene in remain_val_scenes:
        Log.info(f'{scene}: {regi_ratio[scene]}')
        Log.info(f'/mnt/bn/haotongdata/Datasets/scannetpp/data/{scene}/scans/mesh_aligned_0.05.ply')

    


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/splits')
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
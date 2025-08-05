import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm

from lib.utils.pylogger import Log
sys.path.append('.')
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all_scene_path', type=str, default='/mnt/bn/haotongdata/home/linhaotong/workspaces/pl_htcode/txts/scannetpp/regi_iphone_scene_infos.json')
    parser.add_argument('--filter_scene_path', type=str, default='/mnt/bn/haotongdata/home/linhaotong/workspaces/pl_htcode/txts/scannetpp/processed_scene.txt')
    parser.add_argument('--output_path', type=str, default='/mnt/bn/haotongdata/home/linhaotong/workspaces/pl_htcode/txts/scannetpp/iphone_small_waiting_process.txt')
    args = parser.parse_args()
    return args

def main(args):
    filter_scenes = open(args.filter_scene_path).readlines()
    filter_scenes = [scene.strip() for scene in filter_scenes]
    filter_scenes.extend(['1831b3823a', '1a8e0d78c0'])
    all_scenes = {}
    for line in open(args.all_scene_path, 'r').readlines():
        line_info = json.loads(line.strip())
        all_scenes[line_info[0]] = line_info[1]
    waiting_scenes = []
    for scene in all_scenes:
        if scene in filter_scenes: print(scene); continue
        regi_ratio = all_scenes[scene]['regi_ratio']
        if all_scenes[scene]['num_regi_iphone'] < 400 and regi_ratio > 0.5: waiting_scenes.append(scene)
    with open(args.output_path, 'w') as f:
        for scene in waiting_scenes:
            f.write('    "' + scene + '"' + '\n')
    Log.info(f'Saved {args.output_path}')

if __name__ == '__main__':
    args = parse_args()
    main(args)
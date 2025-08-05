import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    return args

def process_one_scene(scene, easy_type, args):
    input_dir = args.input
    output_dir = args.output
    seqs = os.listdir(join(input_dir, scene, easy_type))
    for seq in seqs:
        seq_dir = join(input_dir, scene, easy_type, seq)
        img_len = len(os.listdir(join(seq_dir, 'image_left')))
        img_list = np.random.choice(img_len, 3, replace=False)
        for img_idx in img_list:
            img_name = '{:06d}_left.png'.format(img_idx)
            img_path = join(seq_dir, 'image_left', img_name)
            output_path = join(output_dir, scene, easy_type, seq, img_name)
            dpt_path = join(seq_dir, 'depth_left', '{:06d}_left_depth.npy'.format(img_idx))
            depth = np.load(dpt_path)
            depth_min, depth_max = np.percentile(depth, 0.), np.percentile(depth, 100.)  
            output_path =  join(output_dir, '{}_{}_{}_{}_{:.3f}_{:.3f}.jpg'.format(scene, easy_type, seq, img_name[:-4], depth_min, depth_max))
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            os.system('cp {} {}'.format(img_path, output_path))
            print('cp {} {}'.format(img_path, output_path))
            print('depth min: {}, max: {}'.format(depth.min(), depth.max()))

def main(args):
    scenes = ['abandonedfactory', 'abandonedfactory_night', 'amusement', 'carwelding', 'endofworld', 'gascola', 'hospital', 'japanesealley', 'neighborhood', 'ocean', 'office', 'office2', 'oldtown', 'seasidetown', 'seasonsforest', 'seasonsforest_winter', 'soulcity', 'westerndesert']
    for scene in scenes:
        for easy_type in ['Easy', 'Hard']:
            near_fars = process_one_scene(scene, easy_type, args)

if __name__ == '__main__':
    args = parse_args()
    main(args)
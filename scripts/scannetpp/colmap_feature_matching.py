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
    parser.add_argument('--input_dir', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/56a0ec536c/merge_dslr_iphone')
    parser.add_argument('--thresh_views', type=int, default=1000)
    parser.add_argument('--scene', type=str, default=None)
    args = parser.parse_args()
    return args

def main(args):
    root_dir = args.input_dir
    if args.scene is not None: root_dir = root_dir.replace('56a0ec536c', args.scene)
    output_dir = join(root_dir, 'colmap')
    
    image_len = len(os.listdir(join(root_dir, 'images/iphone'))) + len(os.listdir(join(root_dir, 'images/dslr')))
    
    if image_len < args.thresh_views: cmd = f'colmap exhaustive_matcher --database_path {output_dir}/database.db --log_to_stderr 0 >{output_dir}/matching.txt'
    else: cmd = f'colmap vocab_tree_matcher --database_path {output_dir}/database.db --VocabTreeMatching.vocab_tree_path /mnt/bn/haotongdata/home/linhaotong/envs/vocab_tree_flickr100K_words256K.bin'
    # Log.info(cmd)
    os.system(cmd)
    scene = '56a0ec536c' if args.scene is None else args.scene
    open('done.txt', 'a').write(scene + '\n')
    # cmd = f'colmap {args.matcher} --database_path {output_dir}/database.db'
    # Log.info('matching started')
    # Log.info(cmd)
    # if args.do_all or args.extract_gpu:
    #     os.system(cmd)
    # Log.info('matching done')
    
    # Log.info('Mapper started')
    # cmd = f'colmap mapper --database_path {output_dir}/database.db --image_path {root_dir}/images --output_path {output_dir}/sparse'
    # Log.info(cmd)
    # if args.do_all or not args.extract_gpu:
    #     os.system(cmd)
    # Log.info('Mapper done')

if __name__ == '__main__':
    args = parse_args()
    main(args)
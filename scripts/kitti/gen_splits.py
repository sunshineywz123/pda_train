import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm

from lib.utils.pylogger import Log
sys.path.append('.')

def main(args):
    # scenes = sorted(os.listdir(args.input + '/raw'))
    split = {'rgb_files': [],
             'depth_files': [],
             'lowres_files': []}
    tag = args.tag
    seqs = sorted(os.listdir(join(args.input, 'data_depth_annotated', tag)))
    
    for seq in tqdm(seqs):
        depth02s = sorted(os.listdir(join(args.input, 'data_depth_annotated', tag, seq, 'proj_depth', 'groundtruth', 'image_02')))
        depth03s = sorted(os.listdir(join(args.input, 'data_depth_annotated', tag, seq, 'proj_depth', 'groundtruth', 'image_03')))
        lowdepth02s = sorted(os.listdir(join(args.input, 'data_depth_velodyne', tag, seq, 'proj_depth', 'velodyne_raw', 'image_02')))
        lowdepth03s = sorted(os.listdir(join(args.input, 'data_depth_velodyne', tag, seq, 'proj_depth', 'velodyne_raw', 'image_03')))
        scene = seq[:10]
        image02s = sorted(os.listdir(join(args.input, 'raw', scene, seq, 'image_02/data')))
        image03s = sorted(os.listdir(join(args.input, 'raw', scene, seq, 'image_03/data')))
        image02s = [f for f in image02s if f in depth02s]
        image03s = [f for f in image03s if f in depth03s]
        try:
            assert len(depth02s) == len(image02s)
            assert len(depth03s) == len(image03s)
            assert len(lowdepth02s) == len(image02s)
            assert len(lowdepth03s) == len(image03s)
        except:
            import ipdb; ipdb.set_trace()
        
        split['rgb_files'].extend([join('raw', scene, seq, 'image_02/data', f) for f in image02s])
        split['rgb_files'].extend([join('raw', scene, seq, 'image_03/data', f) for f in image03s])
        split['depth_files'].extend([join('data_depth_annotated', tag, seq, 'proj_depth', 'groundtruth', 'image_02', f) for f in depth02s])
        split['depth_files'].extend([join('data_depth_annotated', tag, seq, 'proj_depth', 'groundtruth', 'image_03', f) for f in depth03s])
        split['lowres_files'].extend([join('data_depth_velodyne', tag, seq, 'proj_depth', 'velodyne_raw', 'image_02', f) for f in lowdepth02s])
        split['lowres_files'].extend([join('data_depth_velodyne', tag, seq, 'proj_depth', 'velodyne_raw', 'image_03', f) for f in lowdepth03s])
    
    os.makedirs(args.output, exist_ok=True)
    import json 
    output = join(args.output, f'{tag}_split.json')
    with open(output, 'w') as f:
        json.dump(split, f)
    Log.info(f'Saved {output}')
            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/bn/haotongdata/Datasets/kitti_completion')
    parser.add_argument('--output', type=str, default='data/pl_htcode/processed_datasets/kitti')
    parser.add_argument('--tag', type=str, default='train')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
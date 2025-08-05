import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
import json

from lib.utils.pylogger import Log
sys.path.append('.')
import glob
def get_metadata(scene):
    rgb_files = []
    depth_files = []
    depth_files = sorted(glob.glob(join('/mnt/bn/haotongdata/Datasets/scannetpp/data', scene, 'merge_dslr_iphone', 'depth/v1/npz/*.npz')))
    rgb_files = [depth_file.replace('depth/v1/npz', 'images/iphone').replace('.npz', '.jpg') for depth_file in depth_files]
    low_res_files = [rgb_file.replace('merge_dslr_iphone/images/iphone', 'iphone/depth').replace('.jpg', '.png') for rgb_file in rgb_files]
    try:
        if not os.path.exists(low_res_files[-1]):
            rgb_files = rgb_files[:-1]
            depth_files = depth_files[:-1]
            low_res_files = low_res_files[:-1]
    except:
        import ipdb; ipdb.set_trace()
    return rgb_files, depth_files, low_res_files
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/splits')
    parser.add_argument('--tag', type=str, default='0614')
    args = parser.parse_args()
    return args

def main(args):
    scene_data = json.load(open(join(args.root_dir, 'scene_data_all.json')))
    train_splits = open(join(args.root_dir, 'nvs_sem_train.txt')).readlines()
    test_splits = open(join(args.root_dir, 'nvs_sem_val.txt')).readlines()
    
    all_meta_data = {'rgb_files': [], 'depth_files': [], 'lowres_files': []}
    train_meta_data = {'rgb_files': [], 'depth_files': [], 'lowres_files': []}
    train_scenes = []
    for scene in tqdm(train_splits):
        scene = scene.strip()
        if scene in scene_data and scene_data[scene] == 1 and os.path.exists(join('/mnt/bn/haotongdata/Datasets/scannetpp/data', scene, 'merge_dslr_iphone', 'depth/v1/info.json')):
            train_scenes.append(scene)

    
    test_meta_data = {'rgb_files': [], 'depth_files': [], 'lowres_files': []}
    test_scenes = []
    for scene in tqdm(test_splits):
        scene = scene.strip()
        if scene in scene_data and scene_data[scene] == 1 and os.path.exists(join('/mnt/bn/haotongdata/Datasets/scannetpp/data', scene, 'merge_dslr_iphone', 'depth/v1/info.json')):
            test_scenes.append(scene)
            
    train_orig_scenes = sorted(train_scenes)
    test_orig_scenes = sorted(test_scenes)
    np.random.seed(42)
    test_scenes = list(set(['5f99900f09'] + np.random.choice(test_scenes, 14, replace=False).tolist()))
    train_scenes = train_orig_scenes + [scene for scene in test_orig_scenes if scene not in test_scenes]
    print('5f99900f09' in train_scenes)
    print('5f99900f09' in test_scenes)
    for scene in train_scenes:
        rgb_files, depth_files, lowres_files = get_metadata(scene)
        train_meta_data['rgb_files'].extend(rgb_files)
        train_meta_data['depth_files'].extend(depth_files)
        train_meta_data['lowres_files'].extend(lowres_files)
        
        all_meta_data['rgb_files'].extend(rgb_files)
        all_meta_data['depth_files'].extend(depth_files)
        all_meta_data['lowres_files'].extend(lowres_files)
    for scene in test_scenes:
        rgb_files, depth_files, lowres_files = get_metadata(scene)
        test_meta_data['rgb_files'].extend(rgb_files)
        test_meta_data['depth_files'].extend(depth_files)
        test_meta_data['lowres_files'].extend(lowres_files)
        all_meta_data['rgb_files'].extend(rgb_files)
        all_meta_data['depth_files'].extend(depth_files)
        all_meta_data['lowres_files'].extend(lowres_files)
    
    
    tag = args.tag
    Log.info(f'Train scenes: {len(train_scenes)}')
    Log.info('Train rgb files: {}'.format(len(train_meta_data['rgb_files'])))
    Log.info(f'Test scenes: {len(test_scenes)}')
    Log.info('Test rgb files: {}'.format(len(test_meta_data['rgb_files'])))
    open(join(args.root_dir, f'train_meta_data_{tag}.json'), 'w').write(json.dumps(train_meta_data))
    open(join(args.root_dir, f'test_meta_data_{tag}.json'), 'w').write(json.dumps(test_meta_data))
    open(join(args.root_dir, f'all_meta_data_{tag}.json'), 'w').write(json.dumps(all_meta_data))
    open(join(args.root_dir, f'train_scenes_{tag}.txt'), 'w').write('\n'.join(train_scenes))
    open(join(args.root_dir, f'test_scenes_{tag}.txt'), 'w').write('\n'.join(test_scenes))
if __name__ == '__main__':
    args = parse_args()
    main(args)
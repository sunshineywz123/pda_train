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
    # depth_files = sorted(glob.glob(join('/mnt/bn/haotongdata/Datasets/scannetpp/data', scene, 'merge_dslr_iphone', 'render_depth/*.png')))
    depth_files = sorted(glob.glob(join('/mnt/bn/haotongdata/Datasets/scannetpp/data', scene, 'merge_dslr_iphone', 'depth/v1/npz/*.npz')))
    mesh_depth_files = sorted(glob.glob(join('/mnt/bn/haotongdata/Datasets/scannetpp/data', scene, 'merge_dslr_iphone', 'render_depth/*.png')))

    depth_frames = [os.path.basename(f).split('.')[0] for f in depth_files]
    mesh_depth_frames = [os.path.basename(f).split('.')[0] for f in mesh_depth_files]
    frames = list(set(depth_frames) & set(mesh_depth_frames))
    frames = sorted(frames)

    depth_files = [join('/mnt/bn/haotongdata/Datasets/scannetpp/data', scene, 'merge_dslr_iphone', 'depth/v1/npz', f'{frame}.npz') for frame in frames]
    mesh_depth_files = [join('/mnt/bn/haotongdata/Datasets/scannetpp/data', scene, 'merge_dslr_iphone', 'render_depth', f'{frame}.png') for frame in frames]
    rgb_files = [depth_file.replace('depth/v1/npz', 'images/iphone').replace('.npz', '.jpg') for depth_file in depth_files]
    low_res_files = [rgb_file.replace('merge_dslr_iphone/images/iphone', 'iphone/depth').replace('.jpg', '.png') for rgb_file in rgb_files]
    sem_files = [depth_file.replace('depth/v1/npz', 'segs').replace('.npz', '.png') for depth_file in depth_files]
    try:
        if len(low_res_files) > 0 and not os.path.exists(low_res_files[-1]):
            rgb_files = rgb_files[:-1]
            depth_files = depth_files[:-1]
            low_res_files = low_res_files[:-1]
            sem_files = sem_files[:-1]
            mesh_depth_files = mesh_depth_files[:-1]
    except:
        import ipdb; ipdb.set_trace()
    return rgb_files, depth_files, low_res_files, mesh_depth_files, sem_files
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/splits')
    parser.add_argument('--tag', type=str, default='final_test')
    args = parser.parse_args()
    return args

def main(args):
    scene_data = json.load(open(join(args.root_dir, 'scene_data_all.json')))
    train_splits = open(join(args.root_dir, 'nvs_sem_train.txt')).readlines()
    test_splits = open(join(args.root_dir, 'nvs_sem_val.txt')).readlines()
    
    all_meta_data = {'rgb_files': [], 'depth_files': [], 'lowres_files': [], 'mesh_depth_files': []}
    train_meta_data = {'rgb_files': [], 'depth_files': [], 'lowres_files': [], 'mesh_depth_files': []}
    train_scenes = []
    for scene in tqdm(train_splits):
        scene = scene.strip()
        if scene in scene_data and scene_data[scene] == 1 and os.path.exists(join('/mnt/bn/haotongdata/Datasets/scannetpp/data', scene, 'merge_dslr_iphone', 'depth/v1/info.json')):
            train_scenes.append(scene)

    
    test_meta_data = {'rgb_files': [], 'depth_files': [], 'lowres_files': [], 'mesh_depth_files': []}
    test_scenes = []
    for scene in tqdm(test_splits):
        scene = scene.strip()
        if scene in scene_data and scene_data[scene] == 1 and os.path.exists(join('/mnt/bn/haotongdata/Datasets/scannetpp/data', scene, 'merge_dslr_iphone', 'depth/v1/info.json')):
            test_scenes.append(scene)
            
    train_orig_scenes = sorted(train_scenes)
    test_orig_scenes = sorted(test_scenes)
    test_orig_scenes = ['09c1414f1b', '1ada7a0617', '40aec5fffa', '3e8bba0176', 'acd95847c5', '578511c8a9', '5f99900f09', 'c4c04e6d6c', 'f3d64c30f8', '7bc286c1b6', 'c5439f4607', '286b55a2bf', 'fb5a96b1a2', '7831862f02', '38d58a7a31', 'bde1e479ad', '9071e139d9', '21d970d8de', 'bcd2436daf', 'cc5237fd77']
    # np.random.seed(42)
    # test_scenes = list(set(['5f99900f09'] + np.random.choice(test_scenes, 14, replace=False).tolist()))
    # train_scenes = train_orig_scenes + [scene for scene in test_orig_scenes if scene not in test_scenes]
    
    train_scenes = train_orig_scenes
    test_scenes = test_orig_scenes
    print('5f99900f09' in train_scenes)
    print('5f99900f09' in test_scenes)
    for scene in tqdm(train_scenes):
        rgb_files, depth_files, lowres_files, meshdepth_files, sem_files = get_metadata(scene)
        train_meta_data['rgb_files'].extend(rgb_files)
        train_meta_data['depth_files'].extend(depth_files)
        train_meta_data['lowres_files'].extend(lowres_files)
        train_meta_data['mesh_depth_files'].extend(meshdepth_files)
        # train_meta_data['sem_files'].extend(sem_files)
        
        all_meta_data['rgb_files'].extend(rgb_files)
        all_meta_data['depth_files'].extend(depth_files)
        all_meta_data['lowres_files'].extend(lowres_files)
        all_meta_data['mesh_depth_files'].extend(meshdepth_files)
        # all_meta_data['sem_files'].extend(sem_files)
    for scene in tqdm(test_scenes):
        rgb_files, depth_files, lowres_files, mesh_depth_files, sem_files = get_metadata(scene)
        test_meta_data['rgb_files'].extend(rgb_files)
        test_meta_data['depth_files'].extend(depth_files)
        test_meta_data['lowres_files'].extend(lowres_files)
        test_meta_data['mesh_depth_files'].extend(mesh_depth_files)
        # test_meta_data['sem_files'].extend(sem_files)
        all_meta_data['rgb_files'].extend(rgb_files)
        all_meta_data['depth_files'].extend(depth_files)
        all_meta_data['lowres_files'].extend(lowres_files)
        all_meta_data['mesh_depth_files'].extend(mesh_depth_files)
        # all_meta_data['sem_files'].extend(sem_files)
    
    
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
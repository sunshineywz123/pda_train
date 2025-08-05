import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
from trdparties.colmap.read_write_model import read_model
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scannetpp_path', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data')
    parser.add_argument('--scene', type=str, default='5f99900f09')
    args = parser.parse_args()
    return args

def main(args):
    input_dir = join(args.scannetpp_path, args.scene, 'iphone')
    cams, images, points = read_model(join(input_dir, 'colmap'))
    output_dir = join(input_dir, 'nerfstudio')
    os.makedirs(output_dir, exist_ok=True)
    
    src_dir = join(input_dir, 'rgb')
    tar_dir = join(output_dir, 'images')
    os.system('ln -s {} {}'.format(src_dir, tar_dir))
    
    frames = []
    for k, image in tqdm(images.items()):
        file_path = join('images', image.name)
        colmap_im_id = k
        
        w2c = np.eye(4)
        w2c[:3, :3] = image.qvec2rotmat()
        w2c[:3, 3] = image.tvec
        
        c2w = np.linalg.inv(w2c)
        c2w[0:3, 1:3] *= -1
        c2w = c2w[np.array([1, 0, 2, 3]), :]
        c2w[2, :] *= -1
        transform_matrix = c2w
        
        cam_id = image.camera_id
        
        frame = {
            'file_path': file_path,
            'transform_matrix': transform_matrix.tolist(),
            'colmap_im_id': colmap_im_id,
            'camera_model': 'OPENCV',
            'fl_x': cams[cam_id].params[0],
            'fl_y': cams[cam_id].params[1],
            'cx': cams[cam_id].params[2],
            'cy': cams[cam_id].params[3],
            'w': cams[cam_id].width,
            'h': cams[cam_id].height,
            'k1': cams[cam_id].params[4],
            'k2': cams[cam_id].params[5],
            'p1': cams[cam_id].params[6],
            'p2': cams[cam_id].params[7],
            'k3': 0,
        }
        frames.append(frame)
    out = {}
    out['frames'] = frames
    
    applied_transform = np.eye(4)[:3, :]
    applied_transform = applied_transform[np.array([1, 0, 2]), :]
    applied_transform[2, :] *= -1
    out["applied_transform"] = applied_transform.tolist()
    with open(join(f'{output_dir}', 'transforms.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=4)
        

if __name__ == '__main__':
    args = parse_args()
    main(args)
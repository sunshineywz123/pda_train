import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
from lib.utils.pylogger import Log
from trdparties.colmap.database import COLMAPDatabase
from trdparties.colmap.read_write_model import read_model, rotmat2qvec

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scannetpp_dir', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data')
    parser.add_argument('--scene', type=str, default='cc5237fd77')
    args = parser.parse_args()
    return args

def process_one_scene(scene, scannetpp_dir):
    
    root_dir = join(scannetpp_dir, scene, 'iphone')
    colmap_dir = join(root_dir, 'colmap')
    output_dir = join(root_dir, 'colmap_sfm')
    os.makedirs(join(output_dir, 'processed'), exist_ok=True)
    os.makedirs(join(output_dir, 'triangulation'), exist_ok=True)
    
    cams, images, points3D = read_model(join(colmap_dir))
    img_list = sorted([img.name for img in images.values()])
    open(join(output_dir, 'image_list.txt'), 'w').write('\n'.join(img_list))
    
    cmd = f'colmap feature_extractor --database_path {output_dir}/database.db --image_path {root_dir}/rgb --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV --image_list_path {output_dir}/image_list.txt'
    Log.info('Feature extraction started')
    os.system(cmd)
    Log.info('Feature extraction done')
    
    cmd = f'colmap sequential_matcher --database_path {output_dir}/database.db'
    Log.info('Sequential matching started')
    os.system(cmd)
    Log.info('Sequential matching done')
    
    os.system(f'touch {output_dir}/processed/points3D.txt')
    os.system(f'cp {root_dir}/colmap/cameras.txt {output_dir}/processed')
    
    db = COLMAPDatabase.connect(join(output_dir, 'database.db'))
    orig_images = images
    image_name_2_colmap_id = {img.name: img.id for img in orig_images.values()}
    images = list(db.execute('select * from images'))
    data_list = []
    for i, image in enumerate(images):
        image_id = image[0]
        image_name = image[1]
        cam_id = image[2]
        orig_image_id = image_name_2_colmap_id[image_name]
        q = orig_images[orig_image_id].qvec
        t = orig_images[orig_image_id].tvec
        data = [image_id, *q, *t, cam_id, image_name]
        data = [str(_) for _ in data]
        data = ' '.join(data)
        data_list.append(data + '\n')
        
    with open(f'{output_dir}/processed/images.txt', 'w') as f:
        f.write('\n'.join(data_list))
    cmd = f'colmap point_triangulator --database_path {output_dir}/database.db --image_path {root_dir}/rgb --input_path {output_dir}/processed --output_path {output_dir}/triangulation'
    Log.info('Triangulation started')
    os.system(cmd)
    Log.info('Triangulation done')

def main(args):
    process_one_scene(args.scene, args.scannetpp_dir)

if __name__ == '__main__':
    args = parse_args()
    main(args)
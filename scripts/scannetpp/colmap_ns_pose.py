import json
import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
from trdparties.colmap.database import COLMAPDatabase
from trdparties.colmap.read_write_model import read_model, rotmat2qvec
from trdparties.colmap.read_write_model import read_cameras_binary, write_cameras_binary, read_cameras_text, read_images_text, write_images_binary, write_cameras_text
def transform2opencv(c2w):
    c2w = c2w.copy()
    c2w[2, :] *= -1
    c2w = c2w[np.array([1, 0, 2, 3]), :]
    c2w[0:3, 1:3] *= -1
    return c2w

def main(args):
    data_dir = args.input_dir
    scene = args.scene
    output_dir = join(data_dir, scene, 'merge_dslr_iphone/colmap')
    image_path = join(data_dir, scene, 'merge_dslr_iphone/images')
    colmap_orig_dir = join(data_dir, scene, 'merge_dslr_iphone/colmap/sparse/0')
    colmap_model_dir = join(data_dir, scene, 'merge_dslr_iphone/colmap/_optimized')
    colmap_output_dir = join(data_dir, scene, 'merge_dslr_iphone/colmap/optimized')
    transform_path = join(data_dir, scene, 'merge_dslr_iphone', 'nerfstudio/optim_cam/nerfacto/none/exports/transforms_train.json')
    os.makedirs(colmap_model_dir, exist_ok=True)
    os.makedirs(colmap_output_dir, exist_ok=True)
    
    db = COLMAPDatabase.connect(join(output_dir, 'database.db'))
    data_list = []
    transforms = json.load(open(transform_path))
    name2id = {transform['file_path'][7:]: idx for idx, transform in enumerate(transforms)} # len(images/)=7
    os.system(f'touch {colmap_model_dir}/points3D.txt')
    # os.system(f'touch {colmap_model_dir}/points3D.bin')
    
    cameras = read_cameras_binary(f'{colmap_orig_dir}/cameras.bin')
    write_cameras_text(cameras, f'{colmap_model_dir}/cameras.txt')
    # write_cameras_binary(cameras, f'{colmap_model_dir}/cameras.bin')
    
    _, orig_images, _ = read_model(colmap_orig_dir)
        
    for k, image in orig_images.items():
        image_id = image.id
        image_name = image.name
        cam_id = image.camera_id
        # orig_image_id = image_name_2_colmap_id[image_name]
        c2w = np.eye(4)
        c2w[:3, :4] = np.asarray(transforms[name2id[image_name]]['transform']).reshape(3, 4)
        c2w = transform2opencv(c2w)
        ext = np.linalg.inv(c2w)
        q = rotmat2qvec(ext[:3, :3])
        t = ext[:3, 3]
        data = [image_id, *q, *t, cam_id, image_name]
        data = [str(_) for _ in data]
        data = ' '.join(data)
        data_list.append(data + '\n')
    with open(f'{colmap_model_dir}/images.txt', 'w') as f:
        f.write('\n'.join(data_list))
    images = read_images_text(f'{colmap_model_dir}/images.txt')
    cmd = f'colmap point_triangulator --database_path {output_dir}/database.db --image_path {image_path} --input_path {colmap_model_dir} --output_path {colmap_output_dir}'
    print(cmd)
    os.system(cmd)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data')
    parser.add_argument('--scene', type=str, default='6ee2fc1070')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)


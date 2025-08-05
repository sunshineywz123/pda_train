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
    output_dir = args.output_colmap_path
    colmap_output_dir = join(output_dir, 'optimized')
    os.makedirs(colmap_output_dir, exist_ok=True)
    db = COLMAPDatabase.connect(join(output_dir, 'database.db'))
    # images = list(db.execute('select * from images'))
    data_list = []
    
    transforms = json.load(open(join(args.input_cam_json)))
    name2id = {transform['file_path'][7:]: idx for idx, transform in enumerate(transforms)}
    
    root_dir = '/'.join(output_dir.split('/')[:-1])
    
    os.system(f'touch {output_dir}/optimized/points3D.txt')
    os.system(f'touch {output_dir}/optimized/points3D.bin')
    if args.colmap_type == 'TXT': 
        os.system(f'cp {args.orig_colmap_path}/cameras.txt {output_dir}/optimized')
    else:
        # os.system(f'touch {output_dir}/optimized/points3D.bin')
        # os.system(f'cp {args.orig_colmap_path}/cameras.bin {output_dir}/optimized')
        cameras = read_cameras_binary(f'{args.orig_colmap_path}/cameras.bin')
        write_cameras_text(cameras, f'{output_dir}/optimized/cameras.txt')
        write_cameras_binary(cameras, f'{output_dir}/optimized/cameras.bin')
    _, orig_images, _ = read_model(args.orig_colmap_path)
        
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
        
        # q = orig_images[orig_image_id].qvec
        # t = orig_images[orig_image_id].tvec
        data = [image_id, *q, *t, cam_id, image_name]
        data = [str(_) for _ in data]
        data = ' '.join(data)
        data_list.append(data + '\n')
    with open(f'{output_dir}/optimized/images.txt', 'w') as f:
        f.write('\n'.join(data_list))
    images = read_images_text(f'{output_dir}/optimized/images.txt')
    write_images_binary(images, f'{output_dir}/optimized/images.bin')
    # os.makedirs(f'{output_dir}/optimized_triangulation', exist_ok=True)
    # cmd = f'colmap point_triangulator --database_path {output_dir}/database.db --image_path {args.image_path} --input_path {output_dir}/optimized --output_path {output_dir}/optimized_triangulation'
    # print(cmd)
    # os.system(cmd)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_cam_json', type=str, 
                        default='/mnt/bn/haotongdata/home/linhaotong/workspaces/nerfstudio/scannetpp_5f99900f09_nerfacto/unnamed/nerfacto/2024-05-23_215752/exports/transforms_train.json')
    parser.add_argument('--output_colmap_path', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/5f99900f09/iphone/colmap_sfm')
    parser.add_argument('--orig_colmap_path', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/5f99900f09/iphone/colmap')
    parser.add_argument('--image_path', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/5f99900f09/iphone/rgb')
    parser.add_argument('--colmap_type', type=str, default='TXT')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)


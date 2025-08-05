import json
import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm


sys.path.append('.')

from trdparties.colmap.database import COLMAPDatabase
from trdparties.colmap.read_write_model import read_images_binary, read_model, rotmat2qvec, write_cameras_text

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data')
    parser.add_argument('--scene', type=str, default='5f99900f09')
    parser.add_argument('--split', type=str, default='merge_dslr_iphone')
    parser.add_argument('--image_path', type=str, default='images')
    parser.add_argument('--database_path', type=str, default='colmap/database_pinhole.db')
    parser.add_argument('--output_model_path', type=str, default='colmap/sparse_pinhole_arkit_colmap') # 0, 0_created, 0_txt
    parser.add_argument('--input_imu_path', type=str, default='iphone/pose_intrinsic_imu.json')
    parser.add_argument('--input_model_path', type=str, default='colmap/sparse/0')
    args = parser.parse_args()
    return args

def created_model(database_path, imu_path, image_path, image_list_path, output_created_path, output_model_path, input_model_path=None):

    cmd = f'colmap feature_extractor --database_path {database_path} --image_path {image_path} --ImageReader.camera_model PINHOLE --image_list_path {image_list_path}'
    os.system(cmd)

    cmd = f'colmap exhaustive_matcher --database_path {database_path}'
    os.system(cmd)

    db = COLMAPDatabase.connect(database_path)
    db_images = db.execute("SELECT * FROM images")
    name2id = {}
    for k in db_images:
        image_id, image_name, camera_id = k[:3]
        name = image_name.split('/')[1][:-4]
        name2id[name] = [image_id, camera_id, image_name]
    pose_imu = json.load(open(imu_path))

    images = []
    cameras = []

    if input_model_path is not None:
        input_images = read_images_binary(join(input_model_path, 'images.bin'))
        input_poses = {input_images[k].name.split('/')[1].split('.')[0]:[input_images[k].qvec, input_images[k].tvec]  for k in input_images if 'dslr' not in input_images[k].name}
    else:
        input_poses = None

    for name, (image_id, camera_id, image_name) in name2id.items():
        if name not in pose_imu: print(name, 'not in pose_imu'); continue
        pose = pose_imu[name]
        if input_poses is not None:
            if name not in input_poses: print(name); continue
            qvec, tvec = input_poses[name]
        else:
            c2w = np.asarray(pose['aligned_pose'])
            ext = np.linalg.inv(c2w)
            qvec = rotmat2qvec(ext[:3, :3])
            tvec = ext[:3, 3]
        images.append([image_id, qvec[0], qvec[1], qvec[2], qvec[3], tvec[0], tvec[1], tvec[2], camera_id, image_name])
        ixt = np.asarray(pose['intrinsic'])
        cameras.append([camera_id, 'PINHOLE', 1920, 1440, ixt[0, 0], ixt[1, 1], ixt[0, 2], ixt[1, 2]])
    os.makedirs(output_created_path, exist_ok=True)
    with open(join(output_created_path, 'images.txt'), 'w') as f:
        for image in images:
            f.write(' '.join(map(str, image)) + '\n\n')
    with open(join(output_created_path, 'cameras.txt'), 'w') as f:
        for camera in cameras:
            f.write(' '.join(map(str, camera)) + '\n')
    os.system('touch ' + join(output_created_path, 'points3D.txt'))

    cmd = f'colmap point_triangulator --database_path {database_path} --image_path {image_path} --input_path {output_created_path} --output_path {output_model_path}  --Mapper.ba_refine_focal_length 0'
    os.system(cmd)
    # for i in range(3):
    #     cmd = f'colmap bundle_adjuster --input_path {output_model_path} --output_path {output_model_path} --BundleAdjustment.refine_focal_length 0'
    #     os.system(cmd)
    #     cmd = f'colmap point_triangulator --database_path {database_path} --image_path {image_path} --input_path {output_model_path} --output_path {output_model_path} --Mapper.ba_refine_focal_length 0'
    #     os.system(cmd)


def optimize_pose(output_path, data_path, scene, colmap_path):
    cmd = f'ns-train nerfacto --vis tensorboard --steps-per-save 30000 --steps-per-eval-all-images 50000 --steps-per-eval-batch 50000 --steps-per-eval-image 5000 --output_dir {output_path} --timestamp {scene}_nerfacto --data {data_path} colmap --downscale_factor 1 --eval_mode all --colmap-path {colmap_path}'
    os.system(cmd)
    cmd = f'ns-export cameras --output-dir {output_path}/merge_dslr_iphone/nerfacto/{scene}_nerfacto/exports --load-config {output_path}/merge_dslr_iphone/nerfacto/{scene}_nerfacto/config.yml'
    os.system(cmd)


def transform2opencv(c2w):
    c2w = c2w.copy()
    c2w[2, :] *= -1
    c2w = c2w[np.array([1, 0, 2, 3]), :]
    c2w[0:3, 1:3] *= -1
    return c2w

def export_colmap(input_colmap_path, output_colmap_path, transform_path, database_path, image_path):
    temp_colmap_path = output_colmap_path + '_temp'
    os.makedirs(output_colmap_path, exist_ok=True)
    os.makedirs(temp_colmap_path, exist_ok=True)

    cameras_orig, images_orig, _ = read_model(input_colmap_path)

    transforms = json.load(open(transform_path))
    name2id = {'/'.join(transform['file_path'].split('/')[-2:]): idx for idx, transform in enumerate(transforms)}

    data_list = []
    for k, image in images_orig.items():
        image_id = image.id
        image_name = image.name
        cam_id = image.camera_id
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
    with open(f'{temp_colmap_path}/images.txt', 'w') as f:
        f.write('\n'.join(data_list))
        f.write('\n')
    write_cameras_text(cameras_orig, f'{temp_colmap_path}/cameras.txt')
    cmd = f'touch {temp_colmap_path}/points3D.txt'
    os.system(cmd)

    cmd = f'colmap point_triangulator --database_path {database_path} --image_path {image_path} --input_path {temp_colmap_path} --output_path {output_colmap_path} --Mapper.ba_refine_focal_length 0'
    os.system(cmd)
        
def main(args):
    scene_dir = join(args.root_path, args.scene)
    root_dir = join(args.root_path, args.scene, args.split)
    database_path = join(root_dir, args.database_path)
    image_path = join(root_dir, args.image_path)
    image_list_path = join(root_dir, 'colmap/iphone_image_list.txt')
    output_created_path = join(root_dir, args.output_model_path, '0_created')
    output_model_path = join(root_dir, args.output_model_path, '0')
    input_model_path = join(root_dir, args.input_model_path)
    os.makedirs(output_model_path, exist_ok=True)
    os.makedirs(output_created_path, exist_ok=True)
    created_model(database_path, join(scene_dir, args.input_imu_path), image_path, image_list_path, output_created_path, output_model_path, input_model_path)

    output_path = '/mnt/bn/haotongdata/home/linhaotong/workspaces/nerfstudio/nerfstudio/nerfstudio'
    data_path = root_dir
    scene = args.scene
    colmap_path = join(args.output_model_path, '0')
    optimize_pose(output_path, data_path, scene, colmap_path)

    input_colmap_path = join(root_dir, args.output_model_path, '0')
    output_colmap_path = join(root_dir, args.output_model_path + '_optimized')
    transform_path  = join(output_path, 'merge_dslr_iphone/nerfacto', scene + '_nerfacto', 'exports', 'transforms_train.json')
    export_colmap(input_colmap_path, output_colmap_path, transform_path, database_path, image_path)



if __name__ == '__main__':
    args = parse_args()
    main(args)
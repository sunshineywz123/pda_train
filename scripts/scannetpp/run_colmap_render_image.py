import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
import imageio

sys.path.append('.')
from lib.utils.parallel_utils import parallel_execution
from lib.utils.pylogger import Log
from trdparties.colmap.database import COLMAPDatabase
from trdparties.colmap.read_write_model import read_cameras_binary, read_cameras_text, read_images_text


def select_nearest_images(src_dir, tar_dir, tar_msk_dir, query_dir, tar_txt, depth_dir):
    os.system('rm -rf {}'.format(tar_dir))
    os.system('rm -rf {}'.format(tar_msk_dir))
    os.makedirs(tar_dir, exist_ok=True)
    os.makedirs(tar_msk_dir, exist_ok=True)
    src_images = os.listdir(src_dir)
    query_images = os.listdir(query_dir)
    selected_images = []
    for query_image in tqdm(query_images):
        idx = int(query_image.split('.')[0].split('_')[1])
        candidate_image = 'frame_{:06d}.jpg'.format(idx // 10 * 10)
        if candidate_image not in selected_images and candidate_image in src_images:
            selected_images.append(candidate_image)
        candidate_image = 'frame_{:06d}.jpg'.format(idx // 10 * 10 + 10)
        if candidate_image not in selected_images and candidate_image in src_images:
            selected_images.append(candidate_image)
    for selected_image in tqdm(selected_images):
        src_image_path = join(src_dir, selected_image)
        tar_image_path = join(tar_dir, selected_image)
        os.system('ln -s {} {}'.format(src_image_path, tar_image_path))
        src_depth = np.asarray(imageio.imread(join(depth_dir, selected_image.replace('.jpg', '.png'))))
        msk = src_depth != 0
        msk = msk.astype(np.uint8) * 255
        msk_path = join(tar_msk_dir, selected_image + '.png')
        imageio.imwrite(msk_path, msk)
        
    selected_images = [join('render_rgb', selected_image) for selected_image in selected_images]
    open(tar_txt, 'w').write('\n'.join(selected_images))

def run_colmap(data_root):
    database_path = join(data_root, 'merge_dslr_iphone', 'colmap', 'database_render_rgb.db')
    image_path = join(data_root, 'merge_dslr_iphone', 'images')
    list_path = join(data_root, 'merge_dslr_iphone', 'colmap', 'render_rgb_list.txt')
    camera_params = read_cameras_text(join(data_root, 'iphone', 'colmap', 'cameras.txt'))[1].params
    mask_path = join(data_root, 'merge_dslr_iphone', 'masks')
    output_model = join(data_root, 'merge_dslr_iphone', 'colmap', 'sparse_render_rgb')
    os.makedirs(output_model, exist_ok=True)
    cmd = 'colmap feature_extractor --database_path {} --image_path {} --image_list_path {}  --ImageReader.single_camera_per_folder 1 --ImageReader.camera_model OPENCV --ImageReader.mask_path {} --ImageReader.camera_params {},{},{},{},{},{},{},{}'.format(database_path, image_path, list_path, mask_path, *camera_params)
    os.system(cmd)

    list_path = join(data_root, 'merge_dslr_iphone', 'colmap', 'iphone_image_list.txt')
    cmd = 'colmap feature_extractor --database_path {} --image_path {} --image_list_path {}  --ImageReader.single_camera_per_folder 1 --ImageReader.camera_model OPENCV --ImageReader.mask_path {}'.format(database_path, image_path, list_path, mask_path)
    os.system(cmd)

    cmd = 'colmap exhaustive_matcher --database_path {}'.format(database_path)
    os.system(cmd)

    tar_colmap_temp_dir = join(data_root, 'merge_dslr_iphone', 'colmap', 'temp')
    os.makedirs(tar_colmap_temp_dir, exist_ok=True)
    src_txt = join(data_root, 'iphone', 'colmap', 'cameras.txt')
    tar_txt = join(tar_colmap_temp_dir, 'cameras.txt')
    os.system('cp {} {}'.format(src_txt, tar_txt))
    os.system('touch {}/points3D.txt'.format(tar_colmap_temp_dir))
    # orig_images = open(join(data_root, 'iphone', 'colmap', 'images.txt')).readlines()
    orig_images = read_images_text(join(data_root, 'iphone', 'colmap', 'images.txt'))
    name2id = {orig_images[k].name:k for k in orig_images}
    db = COLMAPDatabase.connect(database_path)
    images = db.execute("SELECT * FROM images")
    image_relation = {}
    for image in images:
        image_id, image_name, camera_id, _, _, _, _, _, _, _ = image
        if 'render_rgb' not in image_name:
            continue
        image_relation[os.path.basename(image_name)] = [image_id, image_name, camera_id]

    output_lines = []
    for name in image_relation:
        image_id = image_relation[name][0]
        qvec = orig_images[name2id[name]].qvec
        tvec = orig_images[name2id[name]].tvec
        camera_id = 1
        image_name = image_relation[name][1]
        line = ' '.join([str(image_id), ' '.join(map(str, qvec)), ' '.join(map(str, tvec)), str(camera_id), image_name])
        output_lines.append(line + '\n')
        output_lines.append('\n')

    open(join(tar_colmap_temp_dir, 'images.txt'), 'w').writelines(output_lines)

    Log.info('Temp colmap dir: {}'.format(tar_colmap_temp_dir))

    cmd = 'colmap point_triangulator --database_path {} --image_path {} --input_path {} --output_path {}'.format(database_path, image_path, tar_colmap_temp_dir, output_model)
    os.system(cmd)

    output_model_temp = output_model + '_temp'
    os.makedirs(output_model_temp, exist_ok=True)

    cmd = 'colmap mapper --database_path {} --image_path {} --input_path {} --output_path {}'.format(database_path, image_path, output_model, output_model_temp)
    os.system(cmd)

    cmd = 'colmap model_merger --input_path1 {} --input_path2 {} --output_path {}'.format(output_model, output_model_temp, output_model)
    os.system(cmd)
    os.system('rm -rf {}'.format(output_model_temp))
    os.system('rm -rf {}'.format(tar_colmap_temp_dir))
    Log.info('Output colmap model: {}'.format(output_model))

def main(scene):
    # select images
    scene_root = '/mnt/bn/haotongdata/Datasets/scannetpp/data'
    data_root = join(scene_root, scene)
    src_dir = join(data_root, 'iphone', 'render_rgb')
    depth_dir = join(data_root, 'iphone', 'render_depth')
    tar_dir = join(data_root, 'merge_dslr_iphone', 'images', 'render_rgb')
    query_dir = join(data_root, 'merge_dslr_iphone', 'images', 'iphone')
    tar_txt = join(data_root, 'merge_dslr_iphone', 'colmap', 'render_rgb_list.txt')
    tar_msk_dir = join(data_root, 'merge_dslr_iphone', 'masks', 'render_rgb')
    select_nearest_images(src_dir, tar_dir, tar_msk_dir, query_dir, tar_txt, depth_dir)

    run_colmap(data_root)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_root', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data')
    parser.add_argument('--scene', type=str, default='5f99900f09')
    parser.add_argument('--batch_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    input_dir = '/mnt/bn/haotongdata/Datasets/scannetpp'
    scenes = open(join(input_dir, 'splits', 'nvs_sem_train.txt')).readlines() + open(join(input_dir, 'splits', 'nvs_sem_val.txt')).readlines()
    scenes = sorted([scene.strip() for scene in scenes])
    scenes = scenes[args.batch_id::args.batch_size]
    parallel_execution(
        scenes, 
        action=main,
        num_processes=12,
        print_progress=True
        )
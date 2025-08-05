import os
from os.path import join
import argparse
import sys
import ipdb
import numpy as np
import cv2
from tqdm import tqdm
import shutil
import sys

sys.path.append('.')
from trdparties.colmap.read_write_model import read_images_binary
from lib.utils.parallel_utils import parallel_execution

from pathlib import Path
from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_retrieval,
)

# images = Path("/mnt/bn/haotongdata/Datasets/mycapture_arkit/20240629_statue1")
# outputs = Path("outputs/sfm/")
# sfm_pairs = outputs / "pairs-netvlad.txt"
# sfm_dir = outputs / "sfm_superpoint+superglue"
# retrieval_conf = extract_features.confs["netvlad"]
# feature_conf = extract_features.confs["superpoint_inloc"]
# matcher_conf = match_features.confs["superglue"]
# matcher_conf['model']['weights'] = 'indoor'

# retrieval_path = extract_features.main(retrieval_conf, images, outputs)
# pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=30)
# feature_path = extract_features.main(feature_conf, images, outputs)
# match_path = match_features.main(
#     matcher_conf, sfm_pairs, feature_conf["output"], outputs
# )

# model = reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', type=str, default='/mnt/bn/haotongdata/Datasets/mycapture_arkit')
    parser.add_argument('--scene', type=str, default='20240629_statue1')
    parser.add_argument('--top_percent', type=int, default=15)
    parser.add_argument('--max_interval', type=int, default=60)
    parser.add_argument('--min_interval', type=int, default=6)
    args = parser.parse_args()
    return args

def main(args):
    data_root = join(args.input_data_dir, args.scene)
    os.makedirs(join(data_root, 'colmap/spsg'), exist_ok=True)

    images = Path(join(data_root, 'images'))
    outputs = Path(join(data_root, 'colmap/spsg'))
    sfm_pairs = outputs / "pairs-netvlad.txt"
    sfm_dir = outputs / "sfm_superpoint+superglue"
    retrieval_conf = extract_features.confs["netvlad"]
    # feature_conf = extract_features.confs["superpoint_inloc"]
    feature_conf = extract_features.confs["superpoint_inloc"]
    matcher_conf = match_features.confs["superglue"]
    matcher_conf['model']['weights'] = 'indoor'

    retrieval_path = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=25)
    feature_path = extract_features.main(feature_conf, images, outputs)
    match_path = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], outputs
    )
    model = reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path,
                                camera_mode='PER_FOLDER',
                                image_options={'camera_model': 'OPENCV'})
    
    tar_dir = join(data_root, 'colmap/sparse')
    os.makedirs(tar_dir, exist_ok=True)

    num_imgs, model_paths = [], []
    for model in sorted(os.listdir(join(data_root, 'colmap/spsg', 'sfm_superpoint+superglue', 'models'))):
        model_path = join(data_root, 'colmap/spsg', 'sfm_superpoint+superglue', 'models', model)
        if os.path.exists(join(model_path, 'images.bin')): 
            num_img = len(read_images_binary(join(model_path, 'images.bin')))
            num_imgs.append(num_img)
            model_paths.append(model_path)
        else: 
            num_img = len(read_images_binary(join(data_root, 'colmap/spsg', 'sfm_superpoint+superglue', 'images.bin')))
            num_imgs.append(num_img)
            model_paths.append(join(data_root, 'colmap/spsg', 'sfm_superpoint+superglue'))
    max_model = model_paths[np.argmax(num_imgs)]
    os.system('cp -r {}/images.bin {}'.format(max_model, tar_dir))
    os.system('cp -r {}/cameras.bin {}'.format(max_model, tar_dir))
    os.system('cp -r {}/points3D.bin {}'.format(max_model, tar_dir))

if __name__ == '__main__':
    args = parse_args()
    main(args)

import json
import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm

from trdparties.colmap.read_write_model import rotmat2qvec
sys.path.append('.')

from trdparties.colmap.database import COLMAPDatabase

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/mnt/bn/haotongdata/Datasets/mycapture_arkit')
    parser.add_argument('--scene', type=str, default='room_412')
    parser.add_argument('--database_path', type=str, default='colmap/database.db')
    parser.add_argument('--output_path', type=str, default='colmap/sparse_pinhole_per_cam')
    parser.add_argument('--input_intri_path', type=str, default='raw/intri.json')
    parser.add_argument('--input_trans_path', type=str, default='raw/trans.json')
    args = parser.parse_args()
    return args

def main(args):

    root_path = join(args.root_path, args.scene)
    database_path = join(root_path, args.database_path)
    output_path = join(root_path, args.output_path)
    input_intri_path = join(root_path, args.input_intri_path)
    input_trans_path = join(root_path, args.input_trans_path)
    image_path = join(root_path, 'images')
    created_model_path = join(output_path, 'created_model')
    output_txt_model_path = join(output_path, 'txt_model')

    os.makedirs(created_model_path, exist_ok=True)
    os.makedirs(output_txt_model_path, exist_ok=True)

    db = COLMAPDatabase.connect(database_path)
    db_images = db.execute("SELECT * FROM images")
    name2id = {}
    for k in db_images:
        image_id, image_name, camera_id = k[:3]
        # camera_id = image_id
        name2id[image_name] = [image_id, camera_id, image_name]

    intris = json.load(open(input_intri_path))
    trans = json.load(open(input_trans_path))
    ks = {idx:k for idx, k in enumerate(intris)}

    images = []
    cameras = []

    for name, (image_id, camera_id, image_name) in name2id.items():
        idx = int(name.split('.')[0])
        intri = intris[ks[idx]]
        tran = trans[ks[idx]]

        ixt = np.asarray(json.loads(intri)).reshape(3, 3).T
        c2w = np.asarray(json.loads(tran)).reshape(4, 4).T  
        # c2w[3, 3] = 1.
        # c2w[:3, 2] *= -1.
        ext = np.linalg.inv(c2w)
        qvec = rotmat2qvec(ext[:3, :3])
        tvec = ext[:3, 3]
        # tvec = ext[:3, 3] * 0.001
        images.append([image_id, qvec[0], qvec[1], qvec[2], qvec[3], tvec[0], tvec[1], tvec[2], camera_id, image_name])
        cameras.append([camera_id, 'PINHOLE', 1920, 1440, ixt[0, 0], ixt[1, 1], ixt[0, 2], ixt[1, 2]])

    os.makedirs(created_model_path, exist_ok=True)
    with open(join(created_model_path, 'images.txt'), 'w') as f:
        for image in images:
            f.write(' '.join(map(str, image)) + '\n\n')
    with open(join(created_model_path, 'cameras.txt'), 'w') as f:
        for camera in cameras:
            f.write(' '.join(map(str, camera)) + '\n')
    os.system('touch ' + join(created_model_path, 'points3D.txt'))

    cmd = 'colmap point_triangulator --database_path {} --image_path {} --input_path {} --output_path {}'.format(database_path, image_path, created_model_path, output_path)
    os.system(cmd)

    cmd = 'colmap model_converter --input_path {} --output_path {} --output_type TXT'.format(output_path, output_txt_model_path)
    os.system(cmd)

if __name__ == '__main__':
    args = parse_args()
    main(args)
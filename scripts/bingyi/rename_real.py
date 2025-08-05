import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
import cv2

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

sys.path.append('.')
from trdparties.colmap.database import COLMAPDatabase
from trdparties.colmap.read_write_model import Camera, Image, read_images_text, write_cameras_binary, write_cameras_text, write_images_text, read_cameras_text, write_points3D_text


def main(args):
    images = read_images_text(args.input_image)
    cameras = read_cameras_text(args.input_camera)
    orig_names = [image.name.split('\\')[-1] for image in images.values()]
    db = COLMAPDatabase.connect(args.input_database)
    
    db_images = db.execute("SELECT * FROM images")
    db_cameras = db.execute("SELECT * FROM cameras")
    db_images_id = {}
    name2id = {}
    for db_image in db_images:
        im_id, im_name, cam_id = db_image[:3]
        db_images_id[im_id] = [im_id, im_name, cam_id]
        name2id[im_name] = im_id
    import ipdb; ipdb.set_trace()
        
    new_images = {}
    new_cameras = {}
    for im_id in images:
        image = images[im_id]
        name = images[im_id].name
        camera = cameras[image.camera_id]
        name = name.split('\\')[-1]
        
        im_id = name2id[name]
        cam_id = db_images_id[im_id][2]
        
        new_images[im_id] = Image(
            id=im_id,
            qvec=image.qvec,
            tvec=image.tvec,
            camera_id=image.camera_id,
            name=name,
            xys=np.empty((0, 2), dtype=np.float64),
            point3D_ids=np.empty((0,), dtype=np.int32),
        )
        
        new_cameras[cam_id] = Camera(
            id=cam_id,
            model=camera.model,
            width=camera.width,
            height=camera.height,
            params=camera.params,
        )
    import ipdb; ipdb.set_trace()
    write_images_text(new_images, args.output + '/images.txt')
    write_cameras_text(new_cameras, args.output + '/cameras.txt')
    os.system('touch ' + args.output + '/points3D.txt')
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, default='/mnt/bn/haotongdata/test/real2/Calibration/images.txt')
    parser.add_argument('--input_camera', type=str, default='/mnt/bn/haotongdata/test/real2/Calibration/cameras.txt')
    parser.add_argument('--input_database', type=str, default='/mnt/bn/haotongdata/test/real2/database.db')
    parser.add_argument('--output', type=str, default='/mnt/bn/haotongdata/test/real2/created')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)
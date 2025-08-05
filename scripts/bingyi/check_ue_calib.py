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
from trdparties.colmap.read_write_model import Camera, Image, write_cameras_binary, write_cameras_text, write_images_text


def main(args):
    input_image_dir = '/mnt/bn/haotongdata/test/Beauty'
    output_image_dir = '/mnt/bn/haotongdata/test/colmap/images'
    input_camera_file = '/mnt/bn/haotongdata/test/cameras.txt'
    input_image_file = '/mnt/bn/haotongdata/test/images.txt'
    output_colmap_dir = '/mnt/bn/haotongdata/test/colmap/created'
    db = COLMAPDatabase.connect('/mnt/bn/haotongdata/test/colmap/database.db')
    
    
    input_camera_file = '/mnt/bn/haotongdata/test/synthetic2_fix/cameras.txt'
    input_image_file = '/mnt/bn/haotongdata/test/synthetic2_fix/images.txt'
    output_colmap_dir = '/mnt/bn/haotongdata/test/synthetic2_fix/created_qxyzw'
    os.makedirs(output_colmap_dir, exist_ok=True)
    db = COLMAPDatabase.connect('/mnt/bn/haotongdata/test/synthetic2_fix/database.db')
    images = db.execute("SELECT * FROM images")
    cameras = db.execute("SELECT * FROM cameras")
    name2image_camera_id = {}
    for image in images:
        name2image_camera_id[image[1]] = (image[0], image[2])
    db.close()
    
    colmap_images, colmap_cameras = {}, {}
    
    cameras = [line.strip() for line in open(input_camera_file).readlines()]
    images = [line.strip() for line in open(input_image_file).readlines()]
    name2camera_info = {}
    
    for image, camera in zip(images, cameras):
        name = image.split(' ')[0]
        # qw, qx, qy, qz = map(float, image.split(' ')[2:6])
        qx, qy, qz, qw = map(float, image.split(' ')[2:6])
        tx, ty, tz = map(float, image.split(' ')[6:9]) 
        fx, fy, cx, cy = map(float, camera.split(' ')[4:8])
        width, height = map(int, camera.split(' ')[2:4])
        
        image_id, camera_id = name2image_camera_id[name]
        colmap_cameras[camera_id] = Camera(
            id = camera_id,
            model = 'PINHOLE',
            width = width,
            height = height,
            params = [fx, fy, cx, cy],
        )
        
        colmap_images[image_id] = Image(
            id = image_id,
            name = name,
            camera_id = camera_id,
            qvec = [qw, qx, qy, qz],
            tvec = np.asarray([tx, ty, tz])/1000.,
            xys = np.empty((0, 2), dtype=np.float64),
            point3D_ids = np.empty((0,), dtype=np.int32),
        )
    os.makedirs(output_colmap_dir, exist_ok=True)
    write_cameras_text(colmap_cameras, join(output_colmap_dir, 'cameras.txt'))
    write_images_text(colmap_images, join(output_colmap_dir, 'images.txt'))
    os.system('touch ' + join(output_colmap_dir, 'points3D.txt'))
        
    
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str, default='./')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)
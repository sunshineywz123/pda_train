import os
from os.path import join
import argparse
import shutil
import sys
import numpy as np
from tqdm import tqdm


sys.path.append('.')
from trdparties.colmap.database import COLMAPDatabase
from trdparties.colmap.read_write_model import Image, read_model, write_cameras_binary, write_cameras_text, write_images_text

def main(args):
    image_dir = join(args.input, 'images')
    output_dir = join(args.output, 'frames')

    calib_cameras, calib_images, calib_points3d = read_model(join(args.input, 'colmap/sparse/0'))
    name2imid = {calib_images[imid].name: imid for imid in calib_images}
    name2camid = {calib_images[imid].name: calib_images[imid].camera_id for imid in calib_images}
    # copy images, copy masks
    # generate model
    # colmap triangulate
    for frame_id in range(300):
        for video in os.listdir(image_dir):
            src = join(image_dir, video, f'{frame_id:06d}.jpg')
            dst = join(output_dir, f'{frame_id:06d}', 'images', f'{video}.jpg')
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            # soft link
            os.symlink(src, dst)
            src = join(image_dir.replace('images', 'bgmtv2'), video, f'{frame_id:06d}.jpg')
            dst = join(output_dir, f'{frame_id:06d}', 'masks', f'{video}.jpg.png')
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            # soft link
            os.symlink(src, dst)
        cmd = f'colmap feature_extractor --database_path {output_dir}/{frame_id:06d}/database.db --image_path {output_dir}/{frame_id:06d}/images --ImageReader.single_camera_per_image 1 --ImageReader.camera_model OPENCV --ImageReader.camera_mask_path {output_dir}/{frame_id:06d}/masks'
        os.system(cmd)
        cmd = f'colmap exhaustive_matcher --database_path {output_dir}/{frame_id:06d}/database.db'
        os.system(cmd)
        db = COLMAPDatabase.connect(f'{output_dir}/{frame_id:06d}/database.db')
        cameras = db.execute("SELECT * FROM cameras")
        cameras = [camera for camera in cameras]
        images = db.execute("SELECT * FROM images")
        images = [image for image in images]

        new_images = {}
        new_cameras = {}

        for image in images:
            im_id, name, camera_id = image

            new_images[im_id] = Image(
                id=im_id,
                name=name,
                camera_id=camera_id,
                qvec=calib_images[name2imid[name]].qvec,
                tvec=calib_images[name2imid[name]].tvec,
                xys=np.empty((0, 2), dtype=np.float64),
                point3D_ids=np.empty((0,), dtype=np.int32)
            )
            new_cameras[camera_id] = calib_cameras[name2camid[name]]
        os.makedirs(join(output_dir, f'{frame_id:06d}', 'sparse'), exist_ok=True)
        write_cameras_text(new_cameras, join(output_dir, f'{frame_id:06d}', 'sparse', 'cameras.txt'))
        write_images_text(new_images, join(output_dir, f'{frame_id:06d}', 'sparse', 'images.txt'))
        os.system('touch ' + join(output_dir, f'{frame_id:06d}', 'sparse', 'points3D.txt'))
        os.system(f'colmap point_triangulator --database_path {output_dir}/{frame_id:06d}/database.db --image_path {output_dir}/{frame_id:06d}/images --input_path {output_dir}/{frame_id:06d}/sparse --output_path {output_dir}/{frame_id:06d}/sparse')
        db.close()
        os.system('rm -rf ' + join(output_dir, f'{frame_id:06d}', 'database.db'))
        os.system('rm -rf ' + join(output_dir, f'{frame_id:06d}', 'sparse', 'cameras.txt'))
        os.system('rm -rf ' + join(output_dir, f'{frame_id:06d}', 'sparse', 'images.txt'))
        os.system('rm -rf ' + join(output_dir, f'{frame_id:06d}', 'sparse', 'points3D.txt'))
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/data/home/linhaotong/datasets/mycapture4d/20240728_seq1')
    parser.add_argument('--output', type=str, default='/mnt/data/home/linhaotong/datasets/mycapture4d/20240728_seq1/calib')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)
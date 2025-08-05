import os
from os.path import join
import argparse
import shutil
import sys
import numpy as np
from tqdm import tqdm
import imageio
import open3d as o3d
import cv2


sys.path.append('.')
from lib.utils.pylogger import Log
from trdparties.colmap.database import COLMAPDatabase
from trdparties.colmap.read_write_model import Image, read_model, write_cameras_binary, write_cameras_text, write_images_text

def main(args):
    image_dir = join(args.input, 'images')
    output_dir = join(args.output)
    os.makedirs(output_dir, exist_ok=True)

    calib_cameras, calib_images, calib_points3d = read_model(join(args.input, 'colmap/sparse/0'))
    name2imid = {calib_images[imid].name: imid for imid in calib_images}
    name2camid = {calib_images[imid].name: calib_images[imid].camera_id for imid in calib_images}
    # copy images, copy masks
    # generate model
    # colmap triangulate
    for frame_id in range(1000):
        voxel_size = 0.01
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=voxel_size * 5,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        for video in os.listdir(image_dir):
            depth = np.asarray(imageio.imread(join(image_dir.replace('images', 'depth'), video, f'{frame_id:06d}.png')) / 1000)
            img = np.asarray(imageio.imread(join(image_dir, video, f'{frame_id:06d}.jpg')))
            matting = np.asarray(imageio.imread(join(image_dir.replace('images', 'bgmtv2'), video, f'{frame_id:06d}.jpg'))) > 127
            name = video + '.jpg'
            ext, ixt = np.eye(4), np.eye(3)
            ext[:3, :3] = calib_images[name2imid[name]].qvec2rotmat()
            ext[:3, 3] = calib_images[name2imid[name]].tvec
            ixt[0, 0], ixt[1, 1], ixt[0, 2], ixt[1, 2] = calib_cameras[name2camid[name]].params[:4]
            cam_height, cam_width = calib_cameras[name2camid[name]].height, calib_cameras[name2camid[name]].width
            depth_height, depth_width = depth.shape
            img = cv2.resize(img, (depth_width, depth_height))
            matting = cv2.resize(matting.astype(np.uint8), (depth_width, depth_height), interpolation=cv2.INTER_NEAREST)
            depth[matting == 0] = 100
            ixt[:1] = ixt[:1] * depth_width / cam_width
            ixt[1:2] = ixt[1:2] * depth_height / cam_height
            depth = o3d.geometry.Image(depth.astype(np.float32))
            rgb = o3d.geometry.Image(img.astype(np.uint8))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb, depth, depth_scale=1.0, depth_trunc=5., convert_rgb_to_intensity=False)
            volume.integrate(rgbd, 
                             o3d.camera.PinholeCameraIntrinsic(depth_width, depth_height, ixt[0, 0], ixt[1, 1], ixt[0, 2], ixt[1, 2]), 
                             ext)
        pcd = volume.extract_point_cloud()
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        Log.info('num of points: %d' % len(pcd.points))
        o3d.io.write_point_cloud(join(output_dir, f'{frame_id:06d}.ply'), pcd)
        Log.info(f'frame {frame_id} done: {join(output_dir, f"{frame_id:06d}.ply")}')

        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/data/home/linhaotong/datasets/mycapture4d/20240728_seq1')
    parser.add_argument('--output', type=str, default='/mnt/data/home/linhaotong/datasets/mycapture4d/20240728_seq1/pcd_init/lidar_depth')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)
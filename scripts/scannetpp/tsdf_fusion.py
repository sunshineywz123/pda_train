import json
import os
from os.path import join
import argparse
import sys
import cv2
import imageio
import numpy as np
from tqdm import tqdm
import open3d as o3d
import open3d.core as o3c

from lib.utils.pylogger import Log
from trdparties.colmap.read_write_model import read_cameras_binary, read_images_binary
sys.path.append('.')

def get_intrinsics(cam):
    ixt = np.eye(3).astype(np.float32)
    if cam.model == 'OPENCV':
        ixt[0, 0] = cam.params[0]
        ixt[1, 1] = cam.params[1]
        ixt[0, 2] = cam.params[2]
        ixt[1, 2] = cam.params[3]
    elif cam.model == 'SIMPLE_PINHOLE':
        ixt[0, 0] = cam.params[0]
        ixt[1, 1] = cam.params[0]
        ixt[0, 2] = cam.params[1]
        ixt[1, 2] = cam.params[2]
    elif cam.model == 'PINHOLE':
        ixt[0, 0] = cam.params[0]
        ixt[1, 1] = cam.params[1]
        ixt[0, 2] = cam.params[2]
        ixt[1, 2] = cam.params[3]
    else:
        import ipdb; ipdb.set_trace()
    return ixt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_depth_path', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/56a0ec536c/merge/depth/v1/npz')
    parser.add_argument('--input_colmap_path', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/56a0ec536c/merge/depth/v1/colmap')
    parser.add_argument('--input_rgb_path', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/56a0ec536c/merge/images')
    parser.add_argument('--input_info_json', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/56a0ec536c/merge/depth/v1/info.json')
    parser.add_argument('--output_ply_path', type=str, default='56a_zipnerf_output.ply')
    parser.add_argument('--depth_near', type=float, default=0.2)
    parser.add_argument('--depth_far', type=float, default=3.)
    parser.add_argument('--bound_size', type=float, default=5.)
    parser.add_argument('--voxel_res', type=float, default=1024)
    args = parser.parse_args()
    return args

def main(args):
    # read_depth
    # read_extrinsics, ixtrinsics
    # read_rgb
    cams, images = read_cameras_binary(join(args.input_colmap_path, 'cameras.bin')), read_images_binary(join(args.input_colmap_path, 'images.bin'))
    names = [image.name for k, image in images.items()]
    name2colmap_id = {image.name: k for k, image in images.items()}
    names = sorted(names)
    names = [name for name in names if 'iphone' in name]
    
    info_json = json.load(open(args.input_info_json))
    colmap2metric = info_json['colmap2metric_from_zipnerf']
    metric2colmap = 1/colmap2metric
    
    
    near = args.depth_near * metric2colmap
    far = args.depth_far * metric2colmap
    bound_size = args.bound_size * metric2colmap
    voxel_res = args.voxel_res
    voxel_size = bound_size / voxel_res
    
    print('Creating VoxelBlockGrid') 
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=('tsdf', 'weight', 'color'),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=voxel_size,
        block_resolution=16,
        block_count=100000)
    print('Integrating')
    
    
    # names = names[:10] 
    for idx, name in tqdm(enumerate(names)):
        # if idx % 10 != 0: continue
        img = imageio.imread(join(args.input_rgb_path, name))
        
        ext = np.eye(4)
        ext[:3, :3] = images[name2colmap_id[name]].qvec2rotmat()
        ext[:3, 3] = images[name2colmap_id[name]].tvec
        
        ixt = get_intrinsics(cams[images[name2colmap_id[name]].camera_id])
        k1, k2, p1, p2 = cams[images[name2colmap_id[name]].camera_id].params[4:]
        dist = np.zeros(5)
        dist[:4] = [k1, k2, p1, p2]
        
        depth_path = join(args.input_depth_path, name[7:-4] + '.npz')
        depth = np.load(depth_path)['data'] * metric2colmap
        
        
        img = img.astype(np.float32) / 255
        depth = depth.astype(np.float32)
        
        h, w = img.shape[:2]
        new_ixt, _ = cv2.getOptimalNewCameraMatrix(ixt,dist,(w,h),1,(w,h))
        # img_0 = cv2.undistort(img, ixt, dist)
        img = cv2.undistort(img, ixt, dist, newCameraMatrix=new_ixt)
        depth = cv2.undistort(depth, ixt, dist, newCameraMatrix=new_ixt)
        
        border = 25
        img = img[border:-border, border:-border]
        depth = depth[border:-border, border:-border]
        new_ixt[0, 2] -= border
        new_ixt[1, 2] -= border
        ixt = new_ixt
        
        
        
        extrinsic = o3c.Tensor(ext, o3d.core.Dtype.Float64)
        intrinsic = o3c.Tensor(ixt, o3d.core.Dtype.Float64)
        img = o3d.t.geometry.Image(np.ascontiguousarray(img))
        depth = o3d.t.geometry.Image(np.ascontiguousarray(depth))
        frustum_block_coords = vbg.compute_unique_block_coordinates(depth, intrinsic, extrinsic, 1., far)
        
        vbg.integrate(frustum_block_coords, depth, img,
            intrinsic, intrinsic, extrinsic,
            1., far)
    Log.info('Extracting mesh...')
    mesh = vbg.extract_triangle_mesh().to_legacy()
    Log.info('Extracting ending.')
    o3d.io.write_triangle_mesh(args.output_ply_path, mesh)
    Log.info(f'Output mesh saved to {args.output_ply_path}')
        
        
        

if __name__ == '__main__':
    args = parse_args()
    main(args)
import os
from os.path import join
import argparse
import sys
import imageio
import numpy as np
from tqdm import tqdm
import open3d as o3d
import open3d.core as o3c

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
    parser.add_argument('--input_pred_path', type=str, default='/mnt/bn/haotongdata/home/linhaotong/projects/zipnerf-pytorch/exp/tat/meetingroom_pinhole_100000/test_preds')
    parser.add_argument('--input_colmap_path', type=str, default='/mnt/bn/haotongdata/Datasets/tat/test_scenes/Meetingroom/dense/optimized')
    parser.add_argument('--input_rgb_path', type=str, default='/mnt/bn/haotongdata/Datasets/tat/test_scenes/Meetingroom/dense/images')
    parser.add_argument('--output_ply_path', type=str, default='meetingroom_output.ply')
    parser.add_argument('--depth_normalize_factor', type=float, default=1.0496705570887432)
    parser.add_argument('--depth_near', type=float, default=0.2)
    parser.add_argument('--depth_far', type=float, default=1.)
    parser.add_argument('--bound_size', type=float, default=3.)
    parser.add_argument('--voxel_res', type=float, default=2048)
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
    
    near = args.depth_near / args.depth_normalize_factor
    far = args.depth_far / args.depth_normalize_factor   
    bound_size = args.bound_size / args.depth_normalize_factor
    voxel_res = args.voxel_res
    voxel_size = bound_size / voxel_res
    
    print('Creating VoxelBlockGrid') 
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=('tsdf', 'weight', 'color'),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=voxel_size,
        block_resolution=16,
        block_count=800000)
    print('Integrating')
    
    for idx, name in tqdm(enumerate(names[:50])):
        img = imageio.imread(join(args.input_rgb_path, name))
        
        ext = np.eye(4)
        ext[:3, :3] = images[name2colmap_id[name]].qvec2rotmat()
        ext[:3, 3] = images[name2colmap_id[name]].tvec
        
        ixt = get_intrinsics(cams[images[name2colmap_id[name]].camera_id])
        
        depth_path = join(args.input_pred_path, f'distance_median_{idx:04d}.npz')
        depth = np.load(depth_path)['data'] / args.depth_normalize_factor
        
        
        intrinsic = o3c.Tensor(ixt, o3d.core.Dtype.Float64)
        extrinsic = o3c.Tensor(ext, o3d.core.Dtype.Float64)
        img = img.astype(np.float32) / 255
        img = o3d.t.geometry.Image(img)
        depth = depth.astype(np.float32)
        depth = o3d.t.geometry.Image(depth)
        
        frustum_block_coords = vbg.compute_unique_block_coordinates(depth, intrinsic, extrinsic, 1., far)
        
        vbg.integrate(frustum_block_coords, depth, img,
            intrinsic, intrinsic, extrinsic,
            1., far)
    mesh = vbg.extract_triangle_mesh().to_legacy()
    o3d.io.write_triangle_mesh(args.output_ply_path, mesh)
        
        
        

if __name__ == '__main__':
    args = parse_args()
    main(args)
import os
from os.path import join
import argparse
import sys
import numpy as np
from sympy import im
from tqdm import tqdm
sys.path.append('.')
from trdparties.colmap.read_write_model import read_images_text, read_model
import imageio
import cv2
import open3d as o3d
import open3d.core as o3c

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--scene', type=str, default='5f99900f09')
    parser.add_argument('--fusion_type', type=str, default='pred', help='pred, gt, lowres')
    parser.add_argument('--res', type=int, default=512)
    args = parser.parse_args()
    return args

def read_depth(depth_path):
    if depth_path.endswith('.npz'): return np.load(depth_path)['data']
    else: return np.asarray(imageio.imread(depth_path) / 1000.)
    
    
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


def main(args):
    scene = args.scene
    
    tag = 'align_pred'
    exp_name = 'marigold'
    
    voxel_size = 10.0 / args.res
    depth_scale=1.0
    depth_max=10.0
    
    workspace = os.environ['workspace']
    data_root = join(workspace, 'datasets/scannetpp')
    pred_depth_dir = join(workspace, 'outputs/mde/{}/results/scannetpp_{}-1024/{}'.format(exp_name, scene, tag))
    # pred_depth_dir = join(workspace, 'outputs/mde/{}/results/scannetpp_{}-1024/{}'.format(exp_name, scene, tag))
    gt_depth_dir = join(data_root, 'data/{}/iphone/render_depth'.format(scene))
    lowres_depth_dir = join(data_root, 'data/{}/iphone/depth'.format(scene))
    rgb_dir = join(data_root, 'data/{}/iphone/rgb'.format(scene))
    colmap_dir = join(data_root, 'data/{}/iphone/colmap'.format(scene))
    mesh_save_path = join(workspace, 'outputs/mde/{}/results/scannetpp_{}-1024/{}_fuse/mesh-{}-{}.obj'.format(exp_name, scene, tag, args.res, args.fusion_type))
    os.makedirs(os.path.dirname(mesh_save_path), exist_ok=True)
    
    
    colmap_cams, colmap_images, _ = read_model(colmap_dir)
    imgname_to_id = {colmap_images[img].name: img for img in colmap_images}
    
    def read_fusion_assets():
        if args.fusion_type == 'pred':
            depth_dir = pred_depth_dir
            depth_names = sorted(os.listdir(depth_dir))
            depth_img_names = [name[7:-4]+'.jpg' for name in depth_names]
        else:
            depth_dir = gt_depth_dir 
            depth_names = sorted(os.listdir(depth_dir))[::2]
            depth_img_names = [name[:-4]+'.jpg' for name in depth_names]
            depth_dir = lowres_depth_dir if args.fusion_type == 'lowres' else gt_depth_dir
        
        depths = [read_depth(join(depth_dir, depth_name)) for depth_name in depth_names]
        if depths[0].shape[0] < 768:
            depths = [cv2.resize(depth, (1024, 768), interpolation=cv2.INTER_LINEAR) for depth in depths]
        else:
            depths = [cv2.resize(depth, (1024, 768), interpolation=cv2.INTER_NEAREST) for depth in depths]
                
        images, poses, ixts = [], [], []
        for idx, imgname in tqdm(enumerate(depth_img_names), desc='reading assets'):
            # image
            img = imageio.imread(join(rgb_dir, imgname))
            # pose
            ext = np.eye(4)
            ext[:3, :3] = colmap_images[imgname_to_id[imgname]].qvec2rotmat()
            ext[:3, 3] = colmap_images[imgname_to_id[imgname]].tvec
            # ixt
            ixt = get_intrinsics(colmap_cams[colmap_images[imgname_to_id[imgname]].camera_id])
            # TODO: undistortion
            
            # resize
            ixt = ixt.copy()
            if depths[idx].shape[0] != img.shape[0]:
                scale = depths[idx].shape[0] / img.shape[0]
                ixt[:2, :] *= scale
                img = cv2.resize(img, (depths[idx].shape[1], depths[idx].shape[0]), interpolation=cv2.INTER_LINEAR)

            images.append(img)
            poses.append(ext)
            ixts.append(ixt)
        return images, depths, poses, ixts

    images, depths, poses, ixts = read_fusion_assets()
    
    
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=('tsdf', 'weight', 'color'),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=voxel_size,
        block_resolution=16,
        block_count=50000,
        device=o3d.core.Device('CUDA:0'))
    
    for image, depth, pose, ixt in tqdm(zip(images, depths, poses, ixts), desc='fusion process'):
        h, w = image.shape[:2]
        intrinsic = o3c.Tensor(ixt, o3d.core.Dtype.Float64)
        color_intrinsic = depth_intrinsic = intrinsic
    
        ### remove corrupted frames
        val_ratio = (depth > 0.).sum() / (h * w)
        if np.isnan(depth).any() or val_ratio < 0.9: continue

        extrinsic = o3c.Tensor(pose, o3d.core.Dtype.Float64)
        
        img = image.astype(np.float32) / 255
        img = o3d.t.geometry.Image(img).cuda()
    
        depth = depth.astype(np.float32)
        depth = o3d.t.geometry.Image(depth).cuda()
    
        frustum_block_coords = vbg.compute_unique_block_coordinates(
            depth, depth_intrinsic, extrinsic, depth_scale, depth_max)

        vbg.integrate(frustum_block_coords, depth, img,
            depth_intrinsic, color_intrinsic, extrinsic,
            depth_scale, depth_max)
        
    mesh = vbg.extract_triangle_mesh().to_legacy()
    o3d.io.write_triangle_mesh(mesh_save_path, mesh)
    print(f"Mesh saved to {mesh_save_path}")
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
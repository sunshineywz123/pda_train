import dis
import os
from os.path import join
import argparse
import sys
import numpy as np
from sympy import im
from tqdm import tqdm
sys.path.append('.')
from trdparties.colmap.read_write_model import Camera, read_images_text, read_model
import imageio
import cv2
import open3d as o3d
import open3d.core as o3c
import json
from lib.utils.dpt.eval_utils import recover_metric_depth_ransac
from lib.utils.pylogger import Log
from sklearn.neighbors import KDTree

def read_depth(depth_path):
    # if not os.path.exists(depth_path): depth_path = os.path.dirname(depth_path) + '/' + os.path.basename(depth_path)[1:]
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

def get_cv2_distort_params(cam):
    if cam.model == 'OPENCV':
        k1, k2, p1, p2 = cam.params[4:]
        distort_params = np.zeros(5)
        distort_params[:2] = [k1, k2]
        distort_params[2:4] = [p1, p2]
        return np.asarray(distort_params)
    elif cam.model == 'SIMPLE_PINHOLE':
        return np.asarray(cam.params[3:])
    elif cam.model == 'PINHOLE':
        return np.zeros(5)

def nn_correspondance(verts1, verts2):
    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    kdtree = KDTree(verts1)
    distances, indices = kdtree.query(verts2)
    distances = distances.reshape(-1)

    return distances

def evaluate(mesh_pred, mesh_trgt, threshold=.05, down_sample=.02):
    if isinstance(mesh_pred, str):
        mesh_pred = o3d.io.read_triangle_mesh(mesh_pred)
    if isinstance(mesh_trgt, str):
        mesh_trgt = o3d.io.read_triangle_mesh(mesh_trgt)
    pcd_pred = o3d.geometry.PointCloud(mesh_pred.vertices)
    pcd_trgt = o3d.geometry.PointCloud(mesh_trgt.vertices)

    if down_sample:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)

    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    dist1 = nn_correspondance(verts_pred, verts_trgt)
    dist2 = nn_correspondance(verts_trgt, verts_pred)

    precision = np.mean((dist2 < threshold).astype('float'))
    recal = np.mean((dist1 < threshold).astype('float'))
    fscore = 2 * precision * recal / (precision + recal)
    metrics = {
        'Acc': np.mean(dist2),
        'Comp': np.mean(dist1),
        'Prec': precision,
        'Recall': recal,
        'F-score': fscore,
    }
    return metrics

scannetpp_path = '/mnt/bn/haotongdata/Datasets/scannetpp/data'
scannetpp_benchmark_path = '/mnt/bn/haotongdata/home/linhaotong/workspaces/scannetpp_benchmark'

def main(args):
    # config = json.load(open(args.config_json))
    # scene = config['scene_name']
    scene = 'acd95847c5'
    align_methods = ['gt']
    output_dir = join(scannetpp_benchmark_path, scene + '_gt')
    gt_mesh_path = join(scannetpp_path, scene, 'scans', 'mesh_aligned_0.05.ply')
    # gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_path)
    for align_method in align_methods:
        mesh_path = join(output_dir, 'output_align-{}/mesh.ply'.format(align_method))
        # if not os.path.exists(mesh_path):
        #     fuse_mesh(scene, output_dir, align_methods)
        fuse_mesh(scene, output_dir, align_methods)
        metrics = evaluate(mesh_path, gt_mesh_path)
        output_path = join(output_dir, 'output_align-{}/metrics.json'.format(align_method))
        json.dump(metrics, open(output_path, 'w'))
        Log.info(f'Output metrics saved to {output_path}')
    
def fuse_mesh(scene, output_dir, align_methods):
    
    # cameras, rgb, predict_depths, gt_depths 
    # read data
    colmap_dir = join(scannetpp_path, scene, 'iphone', 'colmap')
    colmap_cams, colmap_images, _ = read_model(colmap_dir)
    imgname_to_id = {colmap_images[img].name: img for img in colmap_images}
    ixt = get_intrinsics(colmap_cams[1])
    ixt_orig = ixt.copy()
    rgbs, exts, pred_depths = {}, {}, {}
    gt_depths = {}
    for id, colmap_image in tqdm(colmap_images.items(), desc='reading data for {}'.format(scene)):
        rgbs[colmap_image.name] = np.asarray(imageio.imread(join(scannetpp_path, scene, 'iphone', 'rgb', colmap_image.name))) / 255.
        
        ext = np.eye(4)
        ext[:3, :3] = colmap_image.qvec2rotmat()
        ext[:3, 3] = colmap_image.tvec
        exts[colmap_image.name] = ext
        gt_depths[colmap_image.name] = read_depth(join(scannetpp_path, scene, 'iphone', 'render_depth', colmap_image.name[:-4] + '.png'))
        rgbs[colmap_image.name] = cv2.undistort(rgbs[colmap_image.name], ixt_orig, get_cv2_distort_params(colmap_cams[1]))
        gt_depths[colmap_image.name] = cv2.undistort(gt_depths[colmap_image.name], ixt_orig, get_cv2_distort_params(colmap_cams[1]))
        pred_depths[colmap_image.name] = gt_depths[colmap_image.name]
            
    last_item = colmap_image.name        
    if pred_depths[last_item].shape[0] != gt_depths[last_item].shape[0] or pred_depths[last_item].shape[1] != gt_depths[last_item].shape[1]:
        ixt[:1] *= pred_depths[last_item].shape[1] / gt_depths[last_item].shape[1]
        ixt[1:2] *= pred_depths[last_item].shape[0] / gt_depths[last_item].shape[0]
    
            
    voxel_size = 10.0 / args.vox_res
    depth_scale=1.0
    depth_max=10.0
             
    # TODO: sfm
    for align_method in align_methods:
        is_disparity = False
        
        vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=voxel_size,
            block_resolution=16,
            block_count=50000,
            device=o3d.core.Device('CUDA:0'))
        
        align_depths = {}
        for id, colmap_image in tqdm(colmap_images.items(), desc='aligning data'):
            pred_depth = pred_depths[colmap_image.name]
            gt_depth = gt_depths[colmap_image.name]# if 'gt' in config['align_methods'] else None
            rgb = rgbs[colmap_image.name]
            pose = exts[colmap_image.name]
            if pred_depth.shape[0] != gt_depth.shape[0] or pred_depth.shape[1] != gt_depth.shape[1]:
                gt_depth = cv2.resize(gt_depth, (pred_depth.shape[1], pred_depth.shape[0]), interpolation=cv2.INTER_NEAREST)
                rgb = cv2.resize(rgb, (pred_depth.shape[1], pred_depth.shape[0]), interpolation=cv2.INTER_AREA)
            # depth = recover_metric_depth_ransac(pred_depth, gt_depth, gt_depth>1e-2, disp=is_disparity)
            depth = gt_depth
            intrinsic = o3c.Tensor(ixt, o3d.core.Dtype.Float64)
            color_intrinsic = depth_intrinsic = intrinsic
            h, w = depth.shape
            val_ratio = (depth > 0.).sum() / (h * w)
            if np.isnan(depth).any() or val_ratio < 0.9: continue

            extrinsic = o3c.Tensor(pose, o3d.core.Dtype.Float64)
        
            img = o3d.t.geometry.Image(rgb.astype(np.float32)).cuda()
    
            depth = depth.astype(np.float32)
            depth = o3d.t.geometry.Image(depth).cuda()
    
            frustum_block_coords = vbg.compute_unique_block_coordinates(
                depth, depth_intrinsic, extrinsic, depth_scale, depth_max)

            vbg.integrate(frustum_block_coords, depth, img,
                depth_intrinsic, color_intrinsic, extrinsic,
                depth_scale, depth_max)
        output_path = join(output_dir, 'output_align-{}/mesh.ply'.format(align_method))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mesh = vbg.extract_triangle_mesh().to_legacy()
        o3d.io.write_triangle_mesh(output_path, mesh)
        Log.info(f'Output mesh saved to {output_path}')
            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_json', type=str, default='/mnt/bn/haotongdata/home/linhaotong/workspaces/scannetpp_benchmark/7b6477cb95_depth_anything_v2/config.json')
    parser.add_argument('--vox_res', type=int, default=1024)
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
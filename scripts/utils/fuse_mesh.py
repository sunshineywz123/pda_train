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


from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
degree = 1
poly_features = PolynomialFeatures(degree=degree, include_bias=False)
ransac = RANSACRegressor(max_trials=1000)
model = make_pipeline(poly_features, ransac)


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
        
def get_hws(cam):
    h, w = cam.height, cam.width
    return np.asarray([h, w])

def get_extrinsic(image):
    ext = np.eye(4).astype(np.float32)
    ext[:3, :3] = image.qvec2rotmat()
    ext[:3, 3] = image.tvec
    return ext

def get_sparse_depth(points3d, image, camera):
    # sparse_depth: Nx3 array, uvd
    ixt = get_intrinsics(camera)
    ext = get_extrinsic(image)
    points = np.asarray([points3d[id].xyz for id in image.point3D_ids if id != -1])
    errs = np.asarray([len(points3d[id].image_ids) for id in image.point3D_ids if id != -1])
    
    sparse_depth = points @ ext[:3, :3].T + ext[:3, 3:].T
    sparse_depth = sparse_depth @ ixt.T
    sparse_depth[:, :2] = sparse_depth[:, :2] / sparse_depth[:, 2:]
    hw = get_hws(camera)
    return hw, np.concatenate([sparse_depth, errs[:, None]], axis=1)

def linear_fitting(pred, gt):
    try:
        model.fit(pred[:, None], gt[:, None])
        a, b = model.named_steps['ransacregressor'].estimator_.coef_, model.named_steps['ransacregressor'].estimator_.intercept_
        a = a.item()
        b = b.item()
    except:
        Log.warn('RANSAC failed, using mean scaling')
        a, b = 1, 0
    return a, b
    
def recover_metric_depth_ransac(pred, gt, disp=False, align_method='gt'):
    pred = pred.astype(np.float32)
    if align_method == 'gt':
        if pred.shape[0] != gt.shape[0] or pred.shape[1] != gt.shape[1]:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = gt > 1e-3
        gt = gt.astype(np.float32)
        gt = gt.squeeze()
        pred = pred.squeeze()
        mask = mask.squeeze()
        gt_mask = gt[mask].astype(np.float32)
        pred_mask = pred[mask].astype(np.float32)
    elif align_method == 'sfm':
        hw, sdpt = gt
        h, w = hw
        dpt_h, dpt_w = pred.shape
        u, v = (sdpt[:, 0] * dpt_w / w).astype(np.int32), (sdpt[:, 1] * dpt_h / h).astype(np.int32)
        u_msk = (u >= 0) & (u < dpt_w)
        v_msk = (v >= 0) & (v < dpt_h)
        median = np.median(sdpt[:, 3])
        num_min_points = 50
        thresh_views = sdpt[:, 3].max()
        idx = 30
        while idx > 0:
            idx -= 1
            if (sdpt[:, 3] >= thresh_views).sum() < num_min_points:
                thresh_views -= 1
            elif thresh_views <= 5:
                break
            else:
                break
        if thresh_views <= 5: return None
        print(thresh_views, (sdpt[:, 3] >= thresh_views).sum())
        msk = u_msk & v_msk & (sdpt[:, 3] >= thresh_views)
        gt = sdpt[:, 2]
        gt_mask = gt[msk]
        pred_mask = pred[v[msk], u[msk]]
    
    if disp: 
        gt_mask = np.clip(gt_mask, 1e-3, None)
        gt_mask = 1 / gt_mask
    
    a, b = linear_fitting(pred_mask, gt_mask)
        
    if a > 0:
        pred_metric = a * pred + b
    else:
        pred_mean = np.mean(pred_mask)
        gt_mean = np.mean(gt_mask)
        pred_metric = pred * (gt_mean / pred_mean)
    if disp: pred_metric = 1 / np.clip(pred_metric, 1e-2, None)
    return pred_metric

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
def main(args):
    fuse_mesh(args)
    
    config = json.load(open(args.config_json))
    scene = config['scene_name']
    gt_mesh_path = join(scannetpp_path, scene, 'scans', 'mesh_aligned_0.05.ply')
    for align_method in config['align_methods']:
        mesh_path = join(os.path.dirname(args.config_json), 'output_align-{}/mesh.ply'.format(align_method))
        metrics = evaluate(mesh_path, gt_mesh_path)
        output_path = join(os.path.dirname(args.config_json), 'output_align-{}/metrics.json'.format(align_method))
        json.dump(metrics, open(output_path, 'w'))
        Log.info(f'Output metrics saved to {output_path}')
    
def fuse_mesh(args):
    config = json.load(open(args.config_json))
    scene = config['scene_name']
    depth_dir = join(os.path.dirname(args.config_json), 'depth')
    
    # cameras, rgb, predict_depths, gt_depths 
    # read data
    colmap_dir = join(scannetpp_path, scene, 'iphone', 'colmap')
    colmap_dir = join(scannetpp_path, scene, 'iphone', 'colmap_sfm/triangulation')
    colmap_cams, colmap_images, colmap_points = read_model(colmap_dir)
    imgname_to_id = {colmap_images[img].name: img for img in colmap_images}
    ixt = get_intrinsics(colmap_cams[1])
    ixt_orig = ixt.copy()
    rgbs, exts, pred_depths = {}, {}, {}
    if 'gt' in config['align_methods']: gt_depths = {}
    if 'sfm' in config['align_methods']: sfm_depths = {}
    for id, colmap_image in tqdm(colmap_images.items(), desc='reading data'):
        rgbs[colmap_image.name] = np.asarray(imageio.imread(join(scannetpp_path, scene, 'iphone', 'rgb', colmap_image.name))) / 255.
        
        ext = np.eye(4)
        ext[:3, :3] = colmap_image.qvec2rotmat()
        ext[:3, 3] = colmap_image.tvec
        exts[colmap_image.name] = ext
        
        pred_depths[colmap_image.name] = read_depth(join(depth_dir, colmap_image.name[6:-4] + '.npz'))
        if 'gt' in config['align_methods']:
            gt_depths[colmap_image.name] = read_depth(join(scannetpp_path, scene, 'iphone', 'render_depth', colmap_image.name[:-4] + '.png'))
        if 'sfm' in config['align_methods']:
            sfm_depths[colmap_image.name] = get_sparse_depth(colmap_points, colmap_image, colmap_cams[colmap_image.camera_id])

    # TODO
    last_item = colmap_image.name        
    if pred_depths[last_item].shape[0] != gt_depths[last_item].shape[0] or pred_depths[last_item].shape[1] != gt_depths[last_item].shape[1]:
        ixt[:1] *= pred_depths[last_item].shape[1] / gt_depths[last_item].shape[1]
        ixt[1:2] *= pred_depths[last_item].shape[0] / gt_depths[last_item].shape[0]
    voxel_size = 10.0 / args.vox_res
    depth_scale=1.0
    depth_max=10.0
             
    # TODO: sfm
    for align_method in config['align_methods']:
        is_disparity = config['is_disparity']
        output_path = join(os.path.dirname(args.config_json), 'output_align-{}/mesh.ply'.format(align_method))
        # if os.path.exists(output_path):
        #     continue
        vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=voxel_size,
            block_resolution=16,
            block_count=50000,
            device=o3d.core.Device('CUDA:0'))
        
        if align_method == 'gt':
            align_depths = gt_depths
        elif align_method == 'sfm':
            align_depths = sfm_depths
        
        
        for id, colmap_image in tqdm(colmap_images.items(), desc='aligning data'):
            pred_depth = pred_depths[colmap_image.name]
            gt_depth = gt_depths[colmap_image.name]# if 'gt' in config['align_methods'] else None
            rgb = rgbs[colmap_image.name]
            pose = exts[colmap_image.name]
            if pred_depth.shape[0] != gt_depth.shape[0] or pred_depth.shape[1] != gt_depth.shape[1]:
                rgb = cv2.resize(rgb, (pred_depth.shape[1], pred_depth.shape[0]), interpolation=cv2.INTER_AREA)
            # if align_method == 'gt':
            #     if pred_depth.shape[0] != gt_depth.shape[0] or pred_depth.shape[1] != gt_depth.shape[1]:
            #         gt_depth = cv2.resize(gt_depth, (pred_depth.shape[1], pred_depth.shape[0]), interpolation=cv2.INTER_NEAREST)
            # else:
            #     import ipdb; ipdb.set_trace()
            if align_method != 'metric':
                depth = recover_metric_depth_ransac(pred_depth, align_depths[colmap_image.name], disp=is_disparity, align_method=align_method)
            else:
                depth = pred_depth
            if depth is None:
                continue
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
        output_path = join(os.path.dirname(args.config_json), 'output_align-{}/mesh.ply'.format(align_method))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mesh = vbg.extract_triangle_mesh().to_legacy()
        o3d.io.write_triangle_mesh(output_path, mesh)
        Log.info(f'Output mesh saved to {output_path}')
            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_json', type=str, default='/mnt/bn/haotongdata/home/linhaotong/workspaces/scannetpp_benchmark/7b6477cb95_depth_anything_v2/config.json')
    parser.add_argument('--vox_res', type=int, default=512)
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
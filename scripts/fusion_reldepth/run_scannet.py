import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
import imageio
sys.path.append('.')

from trdparties.colmap.read_write_model import read_model

import open3d as o3d

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth_input', type=str)
    parser.add_argument('--colmap_input', type=str)
    parser.add_argument('--rgb_input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--use_disp', action='store_true')
    parser.add_argument('--step', type=int, default=10)
    args = parser.parse_args()
    return args

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
    errs = np.asarray([points3d[id].error for id in image.point3D_ids if id != -1])
    
    sparse_depth = points @ ext[:3, :3].T + ext[:3, 3:].T
    sparse_depth = sparse_depth @ ixt.T
    sparse_depth[:, :2] = sparse_depth[:, :2] / sparse_depth[:, 2:]
    return np.concatenate([sparse_depth, errs[:, None]], axis=1)

def read_ixts_exts_sdpt(cams, images, points):
    
    # 获得每个按照img_names增序对应的images id
    name2imageid = {img.name:img.id for img in images.values()}
    names = [img.name for img in images.values()]
    names = sorted(names)
    imageids = [name2imageid[name] for name in names]
    
    # ixts, exts, sdpt
    ixts = np.asarray([get_intrinsics(cams[images[imageid].camera_id]) for imageid in imageids])
    hws = np.asarray([get_hws(cams[images[imageid].camera_id]) for imageid in imageids])
    exts = np.asarray([get_extrinsic(images[imageid]) for imageid in imageids])
    
    # sparse_depth: u, v, d
    sdpts = [get_sparse_depth(points, images[imageid], cams[images[imageid].camera_id]) for imageid in imageids]
    return ixts, exts, sdpts, hws
    

def read_monodpts(input_dir):
    npzs = sorted(os.listdir(join(input_dir, 'npys')))
    npzs = [npz for npz in npzs if npz[0] != '.']
    dpts = []
    for npz in npzs:
        npz_path = join(input_dir, 'npys', npz)
        dpt = np.load(npz_path)['dpt']
        dpts.append(dpt)
    return dpts


def create_valid_mask(sdpt, h, w, dpt_h, dpt_w):
    valid_mask = np.zeros((dpt_h, dpt_w)).astype(np.bool_)
    u, v = (sdpt[:, 0] * dpt_w / w).astype(np.int32), (sdpt[:, 1] * dpt_h / h).astype(np.int32)
    valid_mask[v, u] = True
    return valid_mask

def recover_align_dpt(sdpt, dpt, hw, use_disp):
    err = sdpt[:, 3]
    err_min, err_max = np.percentile(err, 5.), np.percentile(err, 50.)
    err_msk = (err < err_max)
    sdpt = sdpt[err_msk][:, :3]
    
    gt = sdpt[:, 2]
    
    h, w = hw
    dpt_h, dpt_w = dpt.shape
    # import ipdb; ipdb.set_trace()
    u, v = (sdpt[:, 0] * dpt_w / w).astype(np.int32), (sdpt[:, 1] * dpt_h / h).astype(np.int32)
    
    u_msk = (u >= 0) & (u < dpt_w)
    v_msk = (v >= 0) & (v < dpt_h)
    msk = u_msk & v_msk
    gt = gt[msk]
    pred = dpt[v[msk], u[msk]]
    # valid_mask = create_valid_mask(sdpt, h, w, dpt_h, dpt_w)
    # pred = dpt[valid_mask]
    if use_disp:
        a, b = np.polyfit(pred, 1/gt, deg=1)
    else:
        a, b = np.polyfit(pred, gt, deg=1)
        
    if a > 0:
        pred_metric = a * dpt + b
    else:
        pred_mean = np.mean(pred)
        gt_mean = np.mean(gt)
        pred_metric = dpt * (gt_mean / pred_mean)
    if use_disp:
        pred_metric = 1 / pred_metric
    print(a, b)
    return pred_metric

def unproject(dpt, ixt, ext):
    h, w = dpt.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    uvd = np.stack([u, v, dpt], axis=-1)
    uvd = uvd.reshape(-1, 3)
    uvd[:, :2] = uvd[:, :2] * uvd[:, 2:]
    ixt_points = uvd @ np.linalg.inv(ixt).T
    ixt_points = np.concatenate([ixt_points, np.ones((ixt_points.shape[0], 1))], axis=1)
    ext_points = ixt_points @ np.linalg.inv(ext).T
    return ext_points[:, :3].reshape(h, w, 3)

def read_imgs(input_dir):
    img_names = sorted(os.listdir(join(input_dir)))
    img_paths = [join(input_dir, img_name) for img_name in img_names]
    imgs = [imageio.imread(img_path) for img_path in img_paths]
    return (np.asarray(imgs) / 255.).astype(np.float32)

def main(args):
    cams, images, points = read_model(args.colmap_input)
    ixts, exts, sdpts, hws = read_ixts_exts_sdpt(cams, images, points)
    imgs = read_imgs(args.rgb_input)
    monodpts = read_monodpts(args.depth_input)
    aligned_dpts = []
    for idx in tqdm(range(len(monodpts))):
        dpt = monodpts[idx]
        aligned_dpt = recover_align_dpt(sdpts[idx], dpt, hws[idx], use_disp=args.use_disp)
        aligned_dpts.append(aligned_dpt)
        
    points_list = []
    colors_list = []
    os.makedirs(args.output, exist_ok=True)
    for i in tqdm(range(0, len(aligned_dpts), args.step)):
        ext = exts[i]
        ixt = ixts[i]
        h, w = hws[i]
        dpt = aligned_dpts[i]
        dpt_h, dpt_w = dpt.shape
        ixt[:1] = ixt[:1] * dpt_w / w
        ixt[1:2] = ixt[1:2] * dpt_h / h
        
        mask = np.ones_like(dpt).astype(np.bool_)
        # mask[1600:] = False
        mask[monodpts[i]<=0.01] = False
        
        points = unproject(dpt, ixt, ext)
        points = points[mask]
        colors = imgs[i][mask].reshape(-1, 3)
        
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        voxel_size = 0.01
        pcd = pcd.voxel_down_sample(voxel_size)
        o3d.io.write_point_cloud(join(args.output, '{:04d}.ply'.format(i)), pcd)
        
        # pcds.append(pcd)
        # rgbs.append(imgs[i].reshape(-1, 3))
    
        
if __name__ == '__main__':
    args = parse_args()
    main(args)
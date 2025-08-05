import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
import imageio
sys.path.append('.')

import h5py

from trdparties.colmap.read_write_model import read_model

import open3d as o3d

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth_input', type=str, default='data/pl_htcode/3d_fusion/hypersim_ai_001_001_cam00/marigold')
    parser.add_argument('--colmap_input', type=str)
    parser.add_argument('--rgb_input', type=str, default='/mnt/remote/D002/Datasets/hypersim/ai_001_001/images/scene_cam_00_final_preview')
    parser.add_argument('--gt_depth_input', type=str, default='/mnt/remote/D002/Datasets/hypersim/ai_001_001/images/scene_cam_00_geometry_hdf5')
    parser.add_argument('--output', type=str, default='data/pl_htcode/3d_fusion/hypersim_ai_001_001_cam00/marigold//3d_fusion')
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
    img_names = [npz.split('.')[1] for npz in npzs]
    return {img_name: dpt for img_name, dpt in zip(img_names, dpts)}


def create_valid_mask(sdpt, h, w, dpt_h, dpt_w):
    valid_mask = np.zeros((dpt_h, dpt_w)).astype(np.bool_)
    u, v = (sdpt[:, 0] * dpt_w / w).astype(np.int32), (sdpt[:, 1] * dpt_h / h).astype(np.int32)
    valid_mask[v, u] = True
    return valid_mask

def recover_align_dpt(gt, dpt, use_disp):
    msk = gt != None
    gt = gt[msk]
    pred = dpt[msk]
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

def unproject_cam(dpt, ixt):
    h, w = dpt.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    uvd = np.stack([u, v, dpt], axis=-1)
    uvd = uvd.reshape(-1, 3)
    uvd[:, :2] = uvd[:, :2] * uvd[:, 2:]
    ixt_points = uvd @ np.linalg.inv(ixt).T
    ixt_points = np.concatenate([ixt_points, np.ones((ixt_points.shape[0], 1))], axis=1)
    return ixt_points[..., :3]


def unproject2(dpt, ixt, ext, ray=False):
    h, w = dpt.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    uvd = np.stack([u, -v, -dpt], axis=-1)
    # uvd = np.stack([u, v, dpt], axis=-1)
    uvd = uvd.reshape(-1, 3)
    if ray:
        uvone = np.concatenate([u, v, np.ones_like(u)], axis=-1)
        import ipdb; ipdb.set_trace()
    else:
        uvd[:, :2] = uvd[:, :2] * uvd[:, 2:]
    ixt_points = uvd @ np.linalg.inv(ixt).T
    ixt_points = np.concatenate([ixt_points, np.ones((ixt_points.shape[0], 1))], axis=1)
    # return ixt_points[..., :3]
    ext_points = ixt_points @ np.linalg.inv(ext).T
    return ext_points[:, :3].reshape(h, w, 3)


def unproject(dpt, ixt, ext, ray=False):
    h, w = dpt.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    # uvd = np.stack([u, v, dpt], axis=-1)
    
    u = (u - ixt[0, 2]) / ixt[0, 0]
    v = (v - ixt[1, 2]) / ixt[1, 1]
    uvd = np.stack([u, -v, -np.ones_like(u)], axis=-1)
    uvd = uvd.reshape(-1, 3)
    uvd = uvd @ np.linalg.inv(ext)[:3, :3].T
    
    points = np.linalg.inv(ext)[:3, 3:].T + uvd.reshape(-1, 3) * dpt.reshape(-1, 1)
    return points

def read_imgs(input_dir):
    img_names = sorted(os.listdir(join(input_dir)))
    img_paths = [join(input_dir, img_name) for img_name in img_names]
    imgs = [imageio.imread(img_path) for img_path in img_paths]
    img_names = [img_name.split('.')[1] for img_name in img_names]
    return {img_name: (img/255.).astype(np.float32) for img_name, img in zip(img_names, imgs)}, img_names

def read_dpts(input_dir):
    img_names = sorted([item for item in os.listdir(join(input_dir)) if item[-4:] == '.npz'])
    dpts = [np.load(join(input_dir, img_name))['data'].astype(np.float32) for img_name in img_names]
    img_names = [img_name.split('.')[1]for img_name in img_names]
    return {img_name: dpt for img_name, dpt in zip(img_names, dpts)}

def read_exts(input_dir):
    orts = h5py.File(join(input_dir, 'camera_keyframe_orientations.hdf5'), 'r')['dataset']
    poss = h5py.File(join(input_dir, 'camera_keyframe_positions.hdf5'), 'r')['dataset']
    exts = []
    for i in range(len(orts)):
        c2w = np.eye(4)
        c2w[:3, :3] = orts[i]
        c2w[:3, 3] = poss[i] * 0.02539999969303608
        ext = np.linalg.inv(c2w)
        exts += [ext]
    return exts

def main(args):
    # cams, images, points = read_model(args.colmap_input)
    # ixts, exts, sdpts, hws = read_ixts_exts_sdpt(cams, images, points)
    dpts = read_dpts(args.gt_depth_input)
    imgs, keys = read_imgs(args.rgb_input)
    exts = read_exts('/mnt/remote/D002/Datasets/hypersim/ai_001_001/_detail/cam_00')
    exts = {key: ext for key, ext in zip(keys, exts)}
    monodpts = read_monodpts(args.depth_input)
    aligned_dpts = {}
    # for idx in tqdm(range(len(monodpts))):
    for k in tqdm(monodpts):
        dpt = monodpts[k]
        aligned_dpt = recover_align_dpt(dpts[k], dpt, use_disp=args.use_disp)
        aligned_dpts[k] = aligned_dpt
    ixt = np.asarray([[886.81, 0, 1024 / 2.], [0, 886.81, 768 / 2.], [0, 0, 1]])
        
    points_list = []
    colors_list = []
    os.makedirs(args.output, exist_ok=True)
    for k in tqdm(aligned_dpts):
        if k not in exts or k not in imgs or k not in dpts:
            continue
        ext = exts[k]
        dpt = aligned_dpts[k]
        dpt_h, dpt_w = dpt.shape
        
        mask = np.ones_like(dpt).astype(np.bool_)
        mask[monodpts[k]<=0.01] = False
        
        points = unproject(dpt, ixt, ext)
        points = points[mask.reshape(-1)]
        colors = imgs[k][mask].reshape(-1, 3)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        voxel_size = 0.01
        pcd = pcd.voxel_down_sample(voxel_size)
        o3d.io.write_point_cloud(join(args.output, '{}.ply'.format(k)), pcd)
        
        # pcds.append(pcd)
        # rgbs.append(imgs[i].reshape(-1, 3))
        
        
    
def debug(args):
    import h5py
    
    hypersim_root = '/mnt/remote/D002/Datasets/hypersim_singlescene_full/ai_001_001'
    geo_dir = 'images/scene_cam_00_geometry_hdf5'
    depth_path = 'frame.0000.depth_meters.hdf5'
    posmap_path = 'frame.0000.position.hdf5'
    
    rot_path = '_detail/cam_00/camera_keyframe_orientations.hdf5'
    pos_path = '_detail/cam_00/camera_keyframe_positions.hdf5'
    unit_scale = 0.02539999969303608
    
    intWidth, intHeight = 1024, 768
    f = 886.81
    
    npyDistance = np.asarray(h5py.File(join(hypersim_root, geo_dir, depth_path), 'r')['dataset'])
    npyImageplaneX = np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth).reshape(1, intWidth).repeat(intHeight, 0).astype(np.float32)[:, :, None]
    npyImageplaneY = np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5, intHeight).reshape(intHeight, 1).repeat(intWidth, 1).astype(np.float32)[:, :, None]
    npyImageplaneZ = np.full([intHeight, intWidth, 1], f, np.float32)
    npyImageplane = np.concatenate([npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)
    npyDepth = npyDistance / np.linalg.norm(npyImageplane, 2, 2) * f
    depth = npyDepth
    
    posmap = np.asarray(h5py.File(join(hypersim_root, geo_dir, posmap_path), 'r')['dataset']) * unit_scale
    
    rot = np.asarray(h5py.File(join(hypersim_root, rot_path), 'r')['dataset'])[0] # c2w, w2c
    pos = np.asarray(h5py.File(join(hypersim_root, pos_path), 'r')['dataset'])[0] # c2w, w2c
    
    # 1. 
    exts = {}
    pattern = '1'
    c2w = np.eye(4)
    c2w[:3, :3] = rot
    c2w[:3, 3] = pos * unit_scale
    ext = np.linalg.inv(c2w)
    exts[pattern] = ext
    
    # 2 
    pattern = '2'
    c2w = np.eye(4)
    c2w[:3, :3] = rot.T
    c2w[:3, 3] = pos * unit_scale
    ext = np.linalg.inv(c2w)
    exts[pattern] = ext
    
    pattern = '3'
    c2w = np.eye(4)
    c2w[:3, :1] = rot[:3, :1]
    c2w[:3, 1:2] = -rot[:3, 1:2]
    c2w[:3, 2:3] = -rot[:3, 2:3]
    c2w[:3, 3] = pos * unit_scale
    ext = np.linalg.inv(c2w)
    exts[pattern] = ext
    
        
    pattern = '4'
    c2w = np.eye(4)
    c2w[:3, :1] = -rot[:3, :1]
    c2w[:3, 1:2] = rot[:3, 1:2]
    c2w[:3, 2:3] = rot[:3, 2:3]
    c2w[:3, 3] = pos * unit_scale
    ext = np.linalg.inv(c2w)
    exts[pattern] = ext
    
    pattern = '5'
    c2w = np.eye(4)
    c2w[:3, :1] = rot[:3, :1]
    c2w[:3, 1:2] = rot[:3, 1:2]
    c2w[:3, 2:3] = -rot[:3, 2:3]
    c2w[:3, 3] = pos * unit_scale
    ext = np.linalg.inv(c2w)
    exts[pattern] = ext
        
    pattern = '6'
    c2w = np.eye(4)
    c2w[:3, :1] = -rot[:3, :1]
    c2w[:3, 1:2] = -rot[:3, 1:2]
    c2w[:3, 2:3] = -rot[:3, 2:3]
    c2w[:3, 3] = pos * unit_scale
    ext = np.linalg.inv(c2w)
    exts[pattern] = ext
    
    f = 886.81
    cx, cy = 1024 / 2., 768 / 2.
    ixt = np.asarray([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    # depth_world = unproject(depth, ixt, ext)
    os.makedirs('debug', exist_ok=True)
    for k in exts:
        depth_world = unproject(depth, ixt, exts[k])
        print(k, depth_world[0], posmap[0, 0])
        # print(k, depth_world[1], posmap[0, 0])
        # print(k, depth_world[200, 200], posmap[200, 200])
        # depth_world = unproject_cam(depth, ixt)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.concatenate([depth_world.reshape(-1, 3), posmap.reshape(-1, 3) @ rot.T], axis=0))
        o3d.io.write_point_cloud('debug/{}.ply'.format(k), pcd)
    # import ipdb; ipdb.set_trace()
    
    
def read_depth(npy_path):
    intWidth, intHeight = 1024, 768
    f = 886.81
    npyDistance = np.asarray(h5py.File(npy_path, 'r')['dataset'])
    npyImageplaneX = np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth).reshape(1, intWidth).repeat(intHeight, 0).astype(np.float32)[:, :, None]
    npyImageplaneY = np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5, intHeight).reshape(intHeight, 1).repeat(intWidth, 1).astype(np.float32)[:, :, None]
    npyImageplaneZ = np.full([intHeight, intWidth, 1], f, np.float32)
    npyImageplane = np.concatenate([npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)
    npyDepth = npyDistance / np.linalg.norm(npyImageplane, 2, 2) * f
    depth = npyDepth
    return depth

def debug2(args):
    import h5py
        
    hypersim_root = '/mnt/remote/D002/Datasets/hypersim_singlescene_full/ai_001_001'
    geo_dir = 'images/scene_cam_00_geometry_hdf5'
    rgb_dir = 'images/scene_cam_00_final_preview'
    depth_pattern = 'frame.{:04d}.depth_meters.hdf5'
    posmap_pattern = 'frame.{:04d}.position.hdf5'
    img_patten = 'frame.{:04d}.color.jpg'
    
    
    rot_path = '_detail/cam_00/camera_keyframe_orientations.hdf5'
    pos_path = '_detail/cam_00/camera_keyframe_positions.hdf5'
    rots = np.asarray(h5py.File(join(hypersim_root, rot_path), 'r')['dataset'])
    poss = np.asarray(h5py.File(join(hypersim_root, pos_path), 'r')['dataset'])
    
    
    ixt = np.asarray([[886.81, 0, 1024 / 2.], [0, 886.81, 768 / 2.], [0, 0, 1]])
    
    os.makedirs('debug', exist_ok=True)
    for i in tqdm(range(10)):
        c2w = np.eye(4)
        c2w[:3, :3] = rots[i]
        c2w[:3, 3] = poss[i] * 0.02539999969303608
        ext = np.linalg.inv(c2w)
        depth_path = join(hypersim_root, geo_dir, depth_pattern.format(i))
        depth = read_depth(depth_path)
        img = imageio.imread(join(hypersim_root, rgb_dir, img_patten.format(i)))
        points = unproject(depth, ixt, ext)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
        pcd.colors = o3d.utility.Vector3dVector(img.reshape(-1, 3) / 255.)
        o3d.io.write_point_cloud('debug/{:04d}.ply'.format(i), pcd)
        
if __name__ == '__main__':
    args = parse_args()
    # debug2(args)
    main(args)
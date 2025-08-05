import os, numpy as np, cv2, matplotlib.pyplot as plt, trimesh
from tqdm import tqdm
import open3d as o3d
import open3d.core as o3c
import json
from os.path import join
from PIL import Image
import sys
sys.path.append('.')
from trdparties.colmap.read_write_model import read_model


def recover_align_dpt_mask(result_depth, sdpt, dpt, mask, use_disp=True):
    gt = sdpt
    msk = (gt > 0) & mask
    gt = gt[msk]
    pred = dpt[msk]
    
    if use_disp:
        a, b = np.polyfit(pred, 1/gt, deg=1)
    else:
        a, b = np.polyfit(pred, gt, deg=1)
        
    if a > 0:
        pred_metric = a * pred + b
    else:
        pred_mean = np.mean(pred)
        gt_mean = np.mean(gt)
        pred_metric = pred * (gt_mean / pred_mean)
    if use_disp:
        pred_metric = 1 / pred_metric
    result_depth[msk] = pred_metric 


def recover_align_dpt_withseg(sdpt, dpt, seg, use_disp=True):
    ids = np.unique(seg)
    result_depth = np.zeros_like(dpt)
    for id in ids:
        mask = (seg == id)
        if mask.sum() >= 100:
            try:
                recover_align_dpt_mask(result_depth, sdpt, dpt, mask, use_disp=use_disp)
            except:
                pass
    return result_depth
    gt = sdpt
    msk = (gt > 0)
    gt = gt[msk]
    pred = dpt[msk]
    
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
    return pred_metric


def recover_align_dpt(sdpt, dpt, use_disp=True):
    gt = sdpt
    msk = (gt > 0)
    gt = gt[msk]
    pred = dpt[msk]
    
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
    return pred_metric

def read_monodpts(input_dir):
    npzs = sorted(os.listdir(os.path.join(input_dir, 'npys')))
    npzs = [npz for npz in npzs if npz[0] != '.']
    dpts = dict()
    for npz in tqdm(npzs, desc='reading monodpts'):
        img_id = int(npz.split('.')[0])
        npz_path = os.path.join(input_dir, 'npys', npz)
        dpt = np.load(npz_path)['dpt']
        # dpts.append(dpt)
        dpts[img_id] = dpt
    return dpts


def read_seg(json_file, img_path):
    view_overseg = Image.open(img_path)
    view_overseg = np.asarray(view_overseg).astype(np.int32)
    view_overseg = (view_overseg[..., 0] << 16) | (view_overseg[..., 1] << 8) | view_overseg[..., 2]
    bg = (view_overseg == 256 ** 3 - 1)
    view_overseg[bg] = -1
    
    seg = np.zeros_like(view_overseg) 
    for seggroup in json_file['segGroups']:
        for segment_id in seggroup['segments']:
            seg[view_overseg == segment_id] = seggroup['id']
    return seg
            

def main(scene_id):
    # mono_dpt_dir = '/mnt/remote/D002/home/linhaotong/research_projects/pl_htcode/data/pl_htcode/outputs/mde/mde_upsample_zerosft_modulated_rgb_from_pretrained_linear_sample/results/scannetpp_perscene'
    # mono_dpt_dir = '/mnt/remote/D002/home/linhaotong/research_projects/pl_htcode/data/pl_htcode/outputs/mde/debug_none/results/scannetpp_perscene_192256'
    mono_dpt_dir = '/mnt/remote/D002/home/linhaotong/research_projects/pl_htcode/data/pl_htcode/outputs/mde/debug_marigold/results/scannetpp_perscene_96128'
    colmap_path = '/mnt/remote/D002/home/linhaotong/Datasets/scannetpp_download/data/98b4ec142f/iphone/colmap'
    image_dir = '/mnt/remote/D002/home/linhaotong/Datasets/scannetpp_download/data/98b4ec142f/iphone/rgb'
    
    cams, images, points = read_model(colmap_path) # type: ignore
    
    mesh_save_path = f'{scene_id}-marigold.obj'
    voxel_size = 10.0 / 512
    depth_scale=1.0
    depth_max=10.0
    
    intrinsic = np.eye(3)
    # 1435.4 1441.79 955.6 723.93
    intrinsic[0, 0], intrinsic[1, 1] = 1435.4/2, 1441.79/2
    intrinsic[0, 2], intrinsic[1, 2] = 955.6/2, 723.93/2
    
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=('tsdf', 'weight', 'color'),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=voxel_size,
        block_resolution=16,
        block_count=50000,
        device=o3d.core.Device('CUDA:0'))
    
    
    intrinsic = o3c.Tensor(intrinsic[:3, :3], o3d.core.Dtype.Float64)
    color_intrinsic = depth_intrinsic = intrinsic
    
    for img_id in tqdm(images):
        image = images[img_id]
        depth_path =  join(mono_dpt_dir, f'{scene_id}_{image.name}_align.npz')
        depth = np.load(depth_path)['pred'].astype(np.float32)
        # mono_scale_dpt = mono_scale_dpts[img_id]
        pose = np.eye(4)
        pose[:3, :3] = image.qvec2rotmat()
        pose[:3, 3] = image.tvec
        # extrinsic = .linalg.inv(pose)
        extrinsic = pose
        extrinsic = o3c.Tensor(extrinsic, o3d.core.Dtype.Float64)
        
        img_path = join(image_dir, f'{image.name}')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1024, 768), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255
        img = o3d.t.geometry.Image(img).cuda()
        # depth = mono_scale_dpt.astype(np.float32)
        # sensor_dpt = sensor_dpts[img_id]
        # depth = sensor_dpt.astype(np.float32)
        depth = o3d.t.geometry.Image(depth).cuda()
        frustum_block_coords = vbg.compute_unique_block_coordinates(
            depth, depth_intrinsic, extrinsic, depth_scale, depth_max)
    
        vbg.integrate(frustum_block_coords, depth, img,
            depth_intrinsic, color_intrinsic, extrinsic,
            depth_scale, depth_max)
    
    mesh = vbg.extract_triangle_mesh().to_legacy()
    o3d.io.write_triangle_mesh(mesh_save_path, mesh)
        
    
if __name__ == '__main__':
    scene_id = '98b4ec142f'
    main(scene_id=scene_id)
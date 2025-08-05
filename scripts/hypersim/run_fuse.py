import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
from glob import glob
import imageio
import pandas as pd
import open3d as o3d
import h5py
import open3d.core as o3c
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--type', type=str, default='pred', help='gt, pred, lowres')
    parser.add_argument('--res', type=int, default=1024)
    parser.add_argument('--scene', type=str, default='ai_001_010')
    parser.add_argument('--cam', type=str, default='cam_00')
    args = parser.parse_args()
    return args

def make_pose(cam_pos, cam_ori, scale):
    R = cam_ori
    R_ = np.copy(R)
    R_[:, 1] *= -1
    R_[:, 2] *= -1
    # R_[1] *= -1
    # R_[2] *= -1
    t = cam_pos.T
    t = t.reshape(3, 1)
    # t[1:] *= -1
    assert R_.shape[0] == R_.shape[1] == 3
    assert t.shape[0] == 3 and t.shape[1] == 1
    
    pose = np.eye(4)
    pose[:3, :3] = R_
    pose[:3, 3:] = t * scale
    
    return pose

def main(args):
    # model_dir = 'data/pl_htcode/outputs/mde/marigold/results'
    model_dir = 'data/pl_htcode/outputs/depth_estimation/{}/results'.format(os.environ['exp'])
    scene = args.scene
    cam = args.cam
    tag = 'orig_pred'
    
    
    gt_dpt_path = "data/pl_htcode/processed_datasets/HyperSim/all/{}/images/scene_{}_geometry_hdf5"
    cam_path = "data/pl_htcode/datasets/HyperSim/all/{}/_detail/{}"
    mono_dpt_path = join(model_dir, '{}-{}'.format(scene, cam), tag)
    image_path = "data/pl_htcode/datasets/HyperSim/all/{}/images/scene_{}_final_preview"
    scale_csv_path = 'data/pl_htcode/datasets/HyperSim/all/{}/_detail/metadata_scene.csv'
    camera_parameters_path = 'data/pl_htcode/processed_datasets/HyperSim/metadata_camera_parameters.csv'
    output_path = join(model_dir, '{}-{}'.format(scene, cam), '{}_fuse'.format(tag), 'mesh-{}-{}.obj'.format(args.res, args.type))
    
    
    if args.type == 'pred':
        dpt_frames = sorted(glob(f"{mono_dpt_path.format(scene, cam)}/*.npz"))
    else:
        dpt_frames = sorted(glob(f"{gt_dpt_path.format(scene, cam)}/*.npz"))
    img_frames = sorted(glob(f"{image_path.format(scene, cam)}/*.tonemap.jpg"))
    img_frames_ = []
    for img_frame in img_frames:
        for dpt_frame in dpt_frames:
            if img_frame[82:100] in dpt_frame:
                img_frames_.append(img_frame)
                break
    img_frames = img_frames_
    def read_gt_dpts():
        dpts = []
        for path in tqdm(dpt_frames):
            # path = os.path.join(gt_dpt_path.format(scene, cam), dpt_frame)
            dpt = np.asarray(np.load(path)["data"]).astype(np.float32)
            dpts.append(dpt)
        return dpts

    def read_images():
        images = list()
        for path in tqdm(img_frames):
            # path = os.path.join(image_path.format(scene, cam), img_frame)
            img = np.asarray(imageio.imread(path)).astype(np.float32)
            images.append(img)
        return images
    
    gt_dpts = read_gt_dpts()
    images = read_images()  
    
    if args.type == 'lowres':
        gt_dpts = [cv2.resize(dpt, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_NEAREST) for dpt in gt_dpts]
        images = [cv2.resize(img, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA) for img in images]
        
    h, w = gt_dpts[0].shape
    
    scale = pd.read_csv(scale_csv_path.format(scene)).to_numpy()[0][1]
    df_camera_parameters = pd.read_csv(camera_parameters_path.format(scene), index_col='scene_name')
    
    
    def get_intrinsic(scene, dfs):
        # Create the placeholder for the intrinsic matrix
        K = np.eye(3)

        # Find the camera parameters for the scene
        df = dfs.loc[scene]
        W, H = int(df["settings_output_img_width"]), int(df["settings_output_img_height"])
        M = np.array([[df["M_proj_00"], df["M_proj_01"], df["M_proj_02"], df["M_proj_03"]],
                      [df["M_proj_10"], df["M_proj_11"], df["M_proj_12"], df["M_proj_13"]],
                      [df["M_proj_20"], df["M_proj_21"], df["M_proj_22"], df["M_proj_23"]],
                      [df["M_proj_30"], df["M_proj_31"], df["M_proj_32"], df["M_proj_33"]]])
    
        # Fill the intrinsic matrix
        K[0, 0] = M[0, 0] * W / 2
        K[0, 2] = -(M[0, 2] * W - W) / 2
        K[1, 1] = M[1, 1] * H / 2
        K[1, 2] = (M[1, 2] * H + H) / 2
        return K, H, W
    
    K, h, w = get_intrinsic(scene, df_camera_parameters)
    if args.type == 'lowres':
        K[0, 0] /= 8
        K[1, 1] /= 8
        K[0, 2] /= 8
        K[1, 2] /= 8
        h //= 8
        w //= 8
    camera_keyframe_positions_hdf5_file = join(cam_path.format(scene, cam), 'camera_keyframe_positions.hdf5')
    camera_keyframe_orientations_hdf5_file = join(cam_path.format(scene, cam), 'camera_keyframe_orientations.hdf5')
    with h5py.File(camera_keyframe_positions_hdf5_file, "r") as f:
        camera_keyframe_positions = f["dataset"][:]
    with h5py.File(camera_keyframe_orientations_hdf5_file, "r") as f:
        camera_keyframe_orientations = f["dataset"][:]
        
        
    voxel_size = 20.0 / args.res
    depth_scale = 1.0
    depth_max = 10.0
    mesh_save_path = output_path
    os.makedirs(os.path.dirname(mesh_save_path), exist_ok=True)

    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=('tsdf', 'weight', 'color'),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=voxel_size,
        block_resolution=16,
        block_count=50000,
        device=o3d.core.Device('CUDA:0'))
    
    for i, gt_dpt in tqdm(enumerate(gt_dpts)):
    # for i, mono_dpt in tqdm(enumerate(mono_dpts)):
        img_id = int(img_frames[i].split('/')[-1].split('.')[1])
        assert h, w == gt_dpt.shape
        intrinsic = o3c.Tensor(K, o3d.core.Dtype.Float64)
        color_intrinsic = depth_intrinsic = intrinsic
    
        ### remove corrupted frames
        val_ratio = (gt_dpt > 0.).sum() / (h * w)
        if np.isnan(gt_dpt).any() or val_ratio < 0.9: continue
    
    

        pose = make_pose(camera_keyframe_positions[img_id], camera_keyframe_orientations[img_id], scale)
        extrinsic = np.linalg.inv(pose)
        extrinsic = o3c.Tensor(extrinsic, o3d.core.Dtype.Float64)
        
        img = images[i]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255
        img = o3d.t.geometry.Image(img).cuda()
    
        depth = gt_dpt.astype(np.float32)
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
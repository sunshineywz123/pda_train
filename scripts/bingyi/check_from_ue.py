import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
import cv2
import json
import pandas as pd
from scipy.spatial.transform import Rotation
import dreifus
# from dreifus.matrix import Pose, CameraCoordinateConvention, PoseType

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

sys.path.append('.')
from lib.utils.coordinates.transform import CoordinateSystem
from trdparties.colmap.database import COLMAPDatabase
from trdparties.colmap.read_write_model import Camera, Image, qvec2rotmat, write_cameras_binary, write_cameras_text, write_images_text, rotmat2qvec


def main(args):
    home_dir = '/mnt/bn/haotongdata/test/BD_Test/RenderingOutput'
    home_dir = '/mnt/bn/haotongdata/test/HalfRotation'
    input_camera = join(home_dir, 'CameraPoses.csv')
    input_ixt = join(home_dir, 'CameraRig.json')
    input_database = join(home_dir, 'CameraComponent', 'database.db')
    output_colmap_dir = join(home_dir, 'CameraComponent', 'created')
    os.makedirs(output_colmap_dir, exist_ok=True)
    
    db = COLMAPDatabase.connect(input_database)
    images = db.execute("SELECT * FROM images")
    cameras = db.execute("SELECT * FROM cameras")
    name2image_camera_id = {}
    names = []
    for image in images:
        name2image_camera_id[image[1]] = (image[0], image[2])
        names.append(image[1])
    db.close()
    names = sorted(names)
    
    ixt =  np.asarray(json.load(open(input_ixt))['cameras']['CameraComponent']['intrinsics']).reshape(3, 3)
    width, height = json.load(open(input_ixt))['cameras']['CameraComponent']['sensor_size']
    fx, fy, cx, cy = ixt[0, 0], ixt[1, 1], ixt[0, 2], ixt[1, 2]
    
    camera_file = pd.read_csv(input_camera)
    txs = camera_file['tx'].values
    tys = camera_file['ty'].values
    tzs = camera_file['tz'].values
    qxs = camera_file['qx'].values
    qys = camera_file['qy'].values
    qzs = camera_file['qz'].values
    qws = camera_file['qw'].values
    
    
    
    conv_function = CoordinateSystem.unreal().convert_func(CoordinateSystem.colmap())
    
    colmap_images, colmap_cameras = {}, {}
    
    # M = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    M = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    
    tvecs = []
    pvecs = []
    for idx, name in enumerate(names):
        # import ipdb; ipdb.set_trace()
        idx = int(name.split('.')[1])
        tx, ty, tz = txs[idx], tys[idx], -tzs[idx]
        qx, qy, qz, qw = qxs[idx], qys[idx], -qzs[idx], -qws[idx]
        
        # rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        # rot = M @ rot
        
        # tvec = np.asarray([tx, ty, tz]) / 100.
        # tvec = M @ tvec
        
        # pose = Pose(ext, pose_type=PoseType.CAM_2_WORLD, camera_coordinate_convention=CameraCoordinateConvention.UNREAL)
        
        # pose.change_camera_coordinate_convention(CameraCoordinateConvention.OPEN_CV)
        # pose.change_pose_type(PoseType.WORLD_2_CAM)
        # import ipdb; ipdb.set_trace()
        
        
        
        # c2w = np.eye(4)
        # c2w[:3, :3] = rot
        # c2w[:3, 3] = tvec
        
        # ext = np.linalg.inv(c2w)
        # qx, qy, qz, qw = Rotation.from_matrix(ext[:3, :3]).as_quat()
        # tx, ty, tz = ext[:3, 3]
        
        colmap_image_id = name2image_camera_id[name][0]
        colmap_camera_id = name2image_camera_id[name][1]
        
        # import ipdb; ipdb.set_trace() 
        rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float32)
        # rot = conv_function(rot).astype(np.float32)
        rot = M @ rot @ M.T
        
        tvec = np.asarray([tx, ty, tz]).astype(np.float32) / 100.
        tvec = M @ tvec
        tvecs.append(tvec)
        
        
        c2w = np.eye(4)
        c2w[:3, :3] = rot
        c2w[:3, 3] = tvec
        ext = np.linalg.inv(c2w)
        
        qvec = rotmat2qvec(ext[:3, :3])
        tvec = ext[:3, 3]
        
        # qx, qy, qz, qw = Rotation.from_matrix(rot).as_quat().astype(np.float32)
        # qvec = np.array([qw, qx, qy, qz]).astype(np.float32)
        
        # qvec = np.array([qx, qy, qz, qw]) 
        # print(qvec, tvec)
        # rot = qvec2rotmat(qvec)
        # c2w = np.eye(4)
        # c2w[:3, :3] = rot
        # c2w[:3, 3] = tvec
        
        # ext = np.linalg.inv(c2w)
        # # # ext = c2w
        # qvec = rotmat2qvec(ext[:3, :3])
        # tvec = ext[:3, 3]
        
        colmap_images[colmap_image_id] = Image(
            id = colmap_image_id,
            name = name,
            camera_id = colmap_camera_id,
            qvec = qvec,
            tvec = tvec,
            xys = np.empty((0, 2), dtype=np.float64),
            point3D_ids = np.empty((0,), dtype=np.int32),
        )
        
        colmap_cameras[colmap_camera_id] = Camera(
            id = colmap_camera_id,
            model = 'PINHOLE',
            width = width,
            height = height,
            params = [fx, fy, cx, cy],
        )
        
        ext = np.eye(4)
        ext[:3, :3] = qvec2rotmat(qvec)
        ext[:3, 3] = tvec
        c2w = np.linalg.inv(ext)
        pvecs.append(c2w[:3, 3])
        
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(tvecs))
    o3d.io.write_point_cloud('tvecs.ply', pcd)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pvecs))
    o3d.io.write_point_cloud('pvecs.ply', pcd)
    import ipdb; ipdb.set_trace()
    
    os.makedirs(output_colmap_dir, exist_ok=True)
    write_cameras_text(colmap_cameras, join(output_colmap_dir, 'cameras.txt'))
    write_images_text(colmap_images, join(output_colmap_dir, 'images.txt'))
    os.system('touch ' + join(output_colmap_dir, 'points3D.txt'))
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str, default='./')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)
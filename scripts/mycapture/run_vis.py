import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
import open3d as o3d
import matplotlib.pyplot as plt
import imageio
import cv2

def load_rgbd_image(color_file, depth_file):
    orig_color = imageio.imread(color_file)
    if depth_file.endswith('.png'):
        orig_depth = (np.asarray(imageio.imread(depth_file)) / 1000.).astype(np.float32)
    else:
        orig_depth = np.load(depth_file)['data']
    color = cv2.resize(orig_color, (orig_depth.shape[1], orig_depth.shape[0]))
    color = o3d.geometry.Image(color)
    depth = o3d.geometry.Image(orig_depth)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_scale=1., depth_trunc=5.0, convert_rgb_to_intensity=False)
    return orig_color, orig_depth, rgbd_image

def rgbd_to_point_cloud(rgbd_image, fx, fy, cx, cy, width, height):
    ixt = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, ixt)
    # o3d.visualization.draw_geometries([pcd])
    return pcd

def create_transformation_matrix():
    # 向上移动 0.5m
    translation = [0, 0.5, 0]  
    # 相机向下俯视 15 度
    rotation = o3d.geometry.get_rotation_matrix_from_xyz([-np.deg2rad(15), 0, 0]) 
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform

# def project_to_rgbd(pcd, intrinsics, width, height):
#     fx, fy = intrinsics.get_focal_length()
#     cx, cy = intrinsics.get_principal_point()
#     camera = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
#     new_rgbd = o3d.geometry.RGBDImage.create_from_point_cloud(pcd, camera)
#     return new_rgbd

# def project_to_rgbd(pcd, intrinsics, width, height):
#     # 创建渲染器
#     renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
#     renderer.setup_camera(intrinsics, pcd.get_center(), pcd.get_center() + [0, -1, 0], [0, 0, 1])
    
#     # 设置材质和灯光
#     mat = o3d.visualization.rendering.MaterialRecord()
#     mat.shader = "defaultUnlit"
#     renderer.scene.add_geometry("pcd", pcd, mat)
#     renderer.scene.scene.set_lighting(renderer.scene, "default")
    
#     # 渲染 RGB 图像
#     img = renderer.render_to_image()
#     renderer.delete_scene("pcd")  # 清理
#     return img

def project_to_rgbd(pcd, intrinsics, width, height):
    # Create the renderer
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    
    # Get the center of the point cloud and calculate camera position and look-at vectors
    center = np.array(pcd.get_center())
    camera_pos = center + np.array([0, 0, 1])  # Modify as needed
    look_at = center
    up = np.array([0, 1, 0])  # Assuming Y is up

    # Setup camera using the correct data types and method signature
    focal_length = intrinsics.intrinsic_matrix[0, 0]  # Assuming fx as the focal length
    renderer.setup_camera(focal_length, camera_pos.astype(np.float32), look_at.astype(np.float32), up.astype(np.float32))
    
    # Material and lighting setup
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    renderer.scene.add_geometry("pcd", pcd, mat)
    # renderer.scene.scene.set_lighting(renderer.scene, "default")
    
    # Render the image
    img = renderer.render_to_image()
    # renderer.delete_scene("pcd")
    return img

def transform_point_cloud(pcd, transform):
    pcd.transform(transform)
    return pcd

def main(args):
    depth_path = '/Users/linhaotong/SSHFS/zjuv06/home/linhaotong/datasets/mycapture4d/20240728_seq1/depth_high/orig_pred/31__000000.npz'
    depth_path = '/Users/linhaotong/SSHFS/zjuv06/home/linhaotong/datasets/mycapture4d/20240728_seq1/depth/31/000000.png'
    rgb_path = '/Users/linhaotong/SSHFS/zjuv06/home/linhaotong/datasets/mycapture4d/20240728_seq1/images_undist/31/000000.jpg'
    fx, fy, cx, cy, width, height = [1443.9070088849212, 1444.523031625396, 942.5, 706, 1885, 1412]
    orig_color, orig_depth, rgbd_image = load_rgbd_image(rgb_path, depth_path)
    fx = fx * orig_depth.shape[1] / orig_color.shape[1]
    fy = fy * orig_depth.shape[0] / orig_color.shape[0]
    cx = cx * orig_depth.shape[1] / orig_color.shape[1]
    cy = cy * orig_depth.shape[0] / orig_color.shape[0]
    width = orig_depth.shape[1]
    height = orig_depth.shape[0]
    ixt = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    transform = create_transformation_matrix()
    pcd = rgbd_to_point_cloud(rgbd_image, fx, fy, cx, cy, width, height)
    transformed_pcd = transform_point_cloud(pcd, transform)
    
    fx, fy, cx, cy, width, height = [1443.9070088849212, 1444.523031625396, 942.5, 706, 1885, 1412]
    ixt = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    new_rgbd = project_to_rgbd(transformed_pcd, ixt, width, height)
    plt.imshow(np.asarray(new_rgbd))
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)
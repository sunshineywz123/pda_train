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
        color, depth, depth_scale=1., depth_trunc=6., convert_rgb_to_intensity=False)
    return orig_color, orig_depth, rgbd_image

def rgbd_to_point_cloud(rgbd_image, fx, fy, cx, cy, width, height):
    ixt = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, ixt)
    # o3d.visualization.draw_geometries([pcd])
    return pcd

def create_transformation_matrix():
    # 向右，向上移动 0.5m, 向后， 
    translation = [-1., 0.5, 0.5]  
    # 相机向下俯视 15 度
    rotation = o3d.geometry.get_rotation_matrix_from_xyz([-np.deg2rad(-8), -np.deg2rad(-12), 0]) 
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

def process_image(depth_path, rgb_path):
    # depth_path = '/Users/linhaotong/SSHFS/zjuv06/home/linhaotong/datasets/mycapture4d/20240728_seq1/depth_high/orig_pred/31__000000.npz'
    # depth_path = '/Users/linhaotong/SSHFS/zjuv06/home/linhaotong/datasets/mycapture4d/20240728_seq1/depth/31/000000.png'
    # rgb_path = '/Users/linhaotong/SSHFS/zjuv06/home/linhaotong/datasets/mycapture4d/20240728_seq1/images_undist/31/000000.jpg'
    fx, fy, cx, cy, width, height = [1443.9070088849212, 1444.523031625396, 942.5, 706, 1885, 1412]
    orig_color, orig_depth, rgbd_image = load_rgbd_image(rgb_path, depth_path)
    fx = fx * orig_depth.shape[1] / orig_color.shape[1]
    fy = fy * orig_depth.shape[0] / orig_color.shape[0]
    cx = cx * orig_depth.shape[1] / orig_color.shape[1]
    cy = cy * orig_depth.shape[0] / orig_color.shape[0]
    width = orig_depth.shape[1]
    height = orig_depth.shape[0]
    pcd = rgbd_to_point_cloud(rgbd_image, fx, fy, cx, cy, width, height)

    transform = create_transformation_matrix()
    transformed_pcd = transform_point_cloud(pcd, transform)

    ixt = np.eye(3)
    ixt[0, 0] = fx
    ixt[1, 1] = fy
    ixt[0, 2] = cx
    ixt[1, 2] = cy
    rvec = np.zeros((3, 1))
    tvec = np.zeros((3, 1))

    points = np.array(transformed_pcd.points).reshape(-1, 3)
    colors = (np.array(transformed_pcd.colors).reshape(-1, 3) * 255).astype(np.uint8)

    # image_points, _ = cv2.projectPoints(points, rvec, tvec, ixt, np.zeros(5))

    cam_points = points @ ixt.T
    depth = cam_points[:, 2]
    image_points = cam_points[:, :2] / depth[:, None]

    x_coords = image_points[:, 0].astype(int)
    y_coords = image_points[:, 1].astype(int)
    # valid_indices = (x_coords >= 0) & (x_coords < width) & (y_coords >= 0) & (y_coords < height)
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    # image[y_coords[valid_indices], x_coords[valid_indices]] = colors[valid_indices]
    valid_indices = (x_coords >= 0) & (x_coords < width) & (y_coords >= 0) & (y_coords < height)
    valid_x_coords = x_coords[valid_indices]
    valid_y_coords = y_coords[valid_indices]
    valid_colors = colors[valid_indices]
    valid_depths = depth[valid_indices]

    sorted_indices = np.argsort(valid_depths)[::-1]
    sorted_x_coords = valid_x_coords[sorted_indices]
    sorted_y_coords = valid_y_coords[sorted_indices]
    sorted_colors = valid_colors[sorted_indices]
    image[sorted_y_coords, sorted_x_coords] = sorted_colors

    # sorted_indices = np.argsort(valid_depths)
    # sorted_x_coords = valid_x_coords[sorted_indices]
    # sorted_y_coords = valid_y_coords[sorted_indices]
    # sorted_colors = valid_colors[sorted_indices]
    image = cv2.resize(image, (1008, 756))
    return image
    # imageio.imwrite('test.jpg', image)
    # plt.imshow(image)
    # plt.show()
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    return args

def main(input, output, pattern):
    frame_len = 600
    images = []
    # input = '/mnt/bn/haotongdata/Datasets/mycapture_stray_scanner/20240728_seq1'
    # output, pattern = 'lidar.mp4', join(input, 'depth', '31', '{:06d}.png')
    # output, pattern = 'da_metric.mp4', '/mnt/bn/haotongdata/home/linhaotong/workspaces/pl_htcode/outputs/depth_estimation/aug_zipmesh_arkit_official_metric_model/results/20240728_seq1_debug_vis/orig_pred/31__{:06d}.npz'
    # output, pattern = 'da_minmax.mp4', '/mnt/bn/haotongdata/home/linhaotong/workspaces/pl_htcode/outputs/depth_estimation/july_hypersim_baseline0717_minmax_finetuned_model/results/20240728_seq1_debug_vis/orig_pred/31__{:06d}.npz'
    # output, pattern = 'depth.mp4', '/mnt/bn/haotongdata/home/linhaotong/workspaces/pl_htcode/outputs/depth_estimation/july_hypersim_baseline0717_minmax_human/results/20240728_seq1_debug_vis/orig_pred/31__{:06d}.npz'
    # output, pattern = 'depth_hypersim.mp4', '/mnt/bn/haotongdata/home/linhaotong/workspaces/pl_htcode/outputs/depth_estimation/july_hypersim/results/20240728_seq1_debug_vis/orig_pred/31__{:06d}.npz'
    # output, pattern = 'depth_minmax.mp4', '/mnt/bn/haotongdata/home/linhaotong/workspaces/pl_htcode/outputs/depth_estimation/july_hypersim_baseline0717_minmax_new_fixbug_1nodes/results/20240728_seq1_debug_vis/orig_pred/31__{:06d}.npz'
    # output, pattern = 'depth_baseline.mp4', '/mnt/bn/haotongdata/home/linhaotong/workspaces/pl_htcode/outputs/depth_estimation/july_hypersim_baseline0717_new/results/20240728_seq1_debug_vis/orig_pred/31__{:06d}.npz'
    for i in tqdm(range(frame_len)):
        depth_path = pattern.format(i)
        rgb_path = join(input, 'images_undist', '31', '{:06d}.jpg'.format(i))
        image = process_image(depth_path, rgb_path)
        # imageio.imwrite('tmp_{}.jpg'.format(output), image)
        # import ipdb; ipdb.set_trace()
        images.append(image)
    imageio.mimwrite(output, images, fps=15)



if __name__ == '__main__':
    args = parse_args()
    input = '/mnt/bn/haotongdata/Datasets/mycapture_stray_scanner/20240728_seq1'
    outputs = ['lidar.mp4', 'da_metric.mp4', 'da_minmax.mp4']
    patterns = [ join(input, 'depth', '31', '{:06d}.png'), '/mnt/bn/haotongdata/home/linhaotong/workspaces/pl_htcode/outputs/depth_estimation/aug_zipmesh_arkit_official_metric_model/results/20240728_seq1_debug_vis/orig_pred/31__{:06d}.npz', '/mnt/bn/haotongdata/home/linhaotong/workspaces/pl_htcode/outputs/depth_estimation/july_hypersim_baseline0717_minmax_new_fixbug_1nodes/results/20240728_seq1_debug_vis/orig_pred/31__{:06d}.npz']
    outputs = ['da_metric.mp4']
    patterns = ['/mnt/bn/haotongdata/home/linhaotong/workspaces/pl_htcode/outputs/depth_estimation/aug_zipmesh_arkit_official_metric_model/results/20240728_seq1_debug_vis/orig_pred/31__{:06d}.npz']
    for output, parttern in zip(outputs, patterns):
        main(input, output, parttern)
    cmd = 'ffmpeg -i lidar.mp4 -i da_metric.mp4 -i da_minmax.mp4 -filter_complex hstack=inputs=3 output_new_2.mp4'
    os.system(cmd)
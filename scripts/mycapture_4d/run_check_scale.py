import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
import imageio
import cv2



from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from PIL import Image as PILImage
from lib.utils.pylogger import Log
from trdparties.colmap.read_write_model import Image, Point3D, read_model, write_model
sys.path.append('.')


depth_path = '/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/outputs/depth_estimation/july_hypersim_scannetpp_zip_mesh/results/calib5_frame1/orig_pred'
# depth_path = '/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/outputs/depth_estimation/july_hypersim_scannetppmesh/results/calib5_frame1/orig_pred'
# depth_path = '/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/outputs/depth_estimation/july_hypersim/results/calib5_frame1/orig_pred'
# depth_path = '/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/outputs/depth_estimation/july_hypersim_scannetppmesh_nograd/results/calib5_frame1/orig_pred'
depth_path = '/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/outputs/depth_estimation/june_depthanythingmetric_scannetpp_0614_hypersim/results/calib5_frame1/orig_pred'
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/bn/haotongdata/Datasets/mycapture_stray_scanner/calib5')
    parser.add_argument('--colmap_path', type=str, default='colmap/sparse_rp/0_metric')
    parser.add_argument('--depth_conf_path', type=str, default='frame1')
    args = parser.parse_args()
    return args

im2cam = {
    'lidar_cam01/lidar_cam01.jpg': '30',
    'lidar_cam02/lidar_cam02.jpg': '31',
    'lidar_cam03/lidar_cam03.jpg': '32',
    'lidar_cam04/lidar_cam04.jpg': '33',
}

def depth2pcd(depth, conf, ixt, ext):
    height, width = depth.shape
    x = np.arange(0, width)
    y = np.arange(0, height)
    xx, yy = np.meshgrid(x, y)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    zz = depth.reshape(-1)
    cc = conf.reshape(-1)
    mask = (cc >= 0) & (zz < 5.)
    xx = xx[mask]
    yy = yy[mask]
    zz = zz[mask]
    pcd = np.stack([xx, yy, np.ones_like(xx)], axis=1)
    pcd = pcd * zz[:, None]
    pcd = np.dot(pcd, np.linalg.inv(ixt).T)
    pcd = np.concatenate([pcd, np.ones((pcd.shape[0], 1))], axis=1)
    pcd = np.dot(pcd, np.linalg.inv(ext).T)
    return pcd[:, :3]

def main(args):
    colmap_path = join(args.input, args.colmap_path)
    cameras, images, points3D = read_model(colmap_path)
    pcds = []
    colors = []

    for im_id in tqdm(images):
        if 'lidar_cam' not in images[im_id].name:
            continue
        cam_id = im2cam[images[im_id].name]

        ext = np.eye(4)
        ext[:3, :3] = images[im_id].qvec2rotmat()
        ext[:3, 3] = images[im_id].tvec

        ixt = np.eye(3)
        camera = cameras[images[im_id].camera_id]
        ixt[0, 0], ixt[1, 1], ixt[0, 2], ixt[1, 2] = camera.params[:4]

        cam_height, cam_width = cameras[images[im_id].camera_id].height, cameras[images[im_id].camera_id].width
        # depth = imageio.imread(join(args.input, args.depth_conf_path, '{}_depth.png'.format(cam_id)))
        # depth = depth / 1000
        depth = np.load(join(depth_path, '{}_rgb.npz'.format(cam_id)))['data']
        depth_height, depth_width = depth.shape
        conf = np.ones_like(depth) * 2
        rgb = imageio.imread(join(args.input, args.depth_conf_path, '{}_rgb.jpg'.format(cam_id)))
        rgb = cv2.resize(rgb, (depth_width, depth_height), interpolation=cv2.INTER_LINEAR)

        ixt[:1] *= depth_width / cam_width
        ixt[1:2] *= depth_height / cam_height
        
        pcd = depth2pcd(depth, conf, ixt, ext)
        pcds.append(pcd)
        colors.append(rgb.reshape(-1, 3)[(conf.reshape(-1) >= 0) & (depth.reshape(-1) < 5.)])
    
    pcds = np.concatenate(pcds, axis=0)
    colors = np.concatenate(colors, axis=0)
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcds)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255)
    o3d.io.write_point_cloud(join(args.input, 'frame1_upsample_depth.ply'), pcd)

if __name__ == '__main__':
    args = parse_args()
    main(args)
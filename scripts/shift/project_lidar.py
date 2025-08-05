import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm

from lib.utils.parallel_utils import parallel_execution
sys.path.append('.')
import cv2
import open3d as o3d
import glob

def project_front_lidar(lidar_path, jpg_path=None, depth_path=None, output_path=None, translation=np.asarray([0.5, 0., 0.]), min_depth=3.):
    pcd = o3d.io.read_point_cloud(lidar_path)
    world_points = np.asarray(pcd.points)
    cam_points = world_points + translation[None]
    cam_points = cam_points[:, [1, 2, 0]]
    cam_points[:, 1] *= -1

    # img = cv2.imread(jpg_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # img_depth = cv2.imread(depth_path)
    # # depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
    DEPTH_C = np.array(1000.0 / (256 * 256 * 256 - 1), np.float32)
    # depth = (img_depth[:, :, 0] * 256. * 256. + img_depth[:, :, 1] * 256. + img_depth[:, :, 2]) * DEPTH_C
    
    ixt = np.eye(3)
    height, width = 800, 1280
    ixt[0, 0], ixt[1, 1], ixt[0, 2], ixt[1, 2] = 640, 640, 640, 400
    
    # focal_x = focal_y = width / (2 * tan(FoV * np.pi / 360.0))
    
    img_points = cam_points @ ixt.T
    msk = img_points[:, 2] > 0.1
    img_points = img_points[msk]
    img_points[:, :2]= img_points[:, :2] / img_points[:, 2:]
    msk = (img_points[:, 0] >= 0) & (img_points[:, 0] < width) & (img_points[:, 1] >= 0) & (img_points[:, 1] < height) & (img_points[:, 2] > min_depth)
    img_points = img_points[msk]
    x = img_points[:, 0].astype(int)
    y = img_points[:, 1].astype(int)
    
    output_depth = np.zeros((height, width), np.float32)
    output_img = np.zeros((height, width, 3), np.uint8)
    output_depth[y, x] = img_points[:, 2]
    # output_depth[y, x] = depth[y, x]
    output_depth = output_depth / DEPTH_C
    output_img[..., 0] = (output_depth // (256 * 256)).astype(np.uint8)
    output_img[..., 1] = ((output_depth % (256 * 256)) // 256).astype(np.uint8)
    output_img[..., 2] = (output_depth % (256 * 256 * 256)).astype(np.uint8)
    if output_path is None: output_path = lidar_path.replace('_center.ply', '_front.png').replace('/center/', '/front/')
    cv2.imwrite(output_path, output_img)
    # cv2.imwrite('test_gt_depth.png', output_img)
    # cv2.imwrite('gt_depth.png', img_depth)
    # import ipdb; ipdb.set_trace()
    # import ipdb; ipdb.set_trace()
    # depth_points = depth[y, x]
    # import matplotlib.pyplot as plt
    # plt.subplot(131)
    # sparse_depth = np.zeros_like(depth)
    # sparse_depth[y, x] = depth_points
    # sparse_depth = np.clip(sparse_depth / 150, 0, 1)
    # plt.imshow(sparse_depth)
    # # plt.plot(img_points[:, 0], img_points[:, 1], 'r.')
    # plt.axis('off')
    
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(cam_points[msk])
    
    # plt.subplot(132)
    # plt.imshow(img)
    # plt.axis('off')
    
    # plt.subplot(133)
    # plt.imshow(np.clip(depth / 150, 0, 1))
    # plt.tight_layout()
    # plt.savefig('test.jpg', dpi=300)
    # plt.axis('off')
    # import ipdb; ipdb.set_trace()

def main(args):
    jpg_path = '/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/preview/haotongdata/Datasets/shift/discrete/images/val/front/0aee-69fd/00000000_img_front.jpg'
    depth_path = '/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/preview/haotongdata/Datasets/shift/discrete/images/val/front/0aee-69fd/00000000_depth_front.png'
    lidar_path = '/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/preview/haotongdata/Datasets/shift/discrete/images/val/center/0aee-69fd/00000000_lidar_center.ply'
    output_path = '/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/preview/haotongdata/Datasets/shift/discrete/images/val/front/0aee-69fd/00000000_lidar_front.png'
    # project_front_lidar(lidar_path, output_path=output_path)
    root_dir = '/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/preview/haotongdata/Datasets/shift/discrete/images'
    splits = ['train', 'val']
    metas = []
    for split in tqdm(splits):
        scenes = os.listdir(join(root_dir, split, 'front'))
        for scene in tqdm(scenes):
            imgs = glob.glob(join(root_dir, split, 'front', scene, '*_img_front.jpg'))
            lidars = glob.glob(join(root_dir, split, 'center', scene, '*_lidar_center.ply'))
            assert len(imgs) == len(lidars), f'{scene} {len(imgs)} {len(lidars)}'
            metas.extend(lidars)
    parallel_execution(metas, 
                       action=project_front_lidar,
                       num_processes=32,
                       print_progress=True)

    
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/bn/haotongdata/Datasets/DyDToF')
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
import trimesh
import tyro
import os 
from os.path import join
from tqdm.auto import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio
import sys 
sys.path.append('.')
from lib.utils.geo_utils import depth2pcd
import open3d as o3d

def validate_align(
    lidar_path: str = '/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/datasets/waymo/226/lidar_depth/000000_0.npy',
    depth_path: str = '/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/outputs/depth_estimation/aug_shift_comp_baseline_new_lidar_precomp_fov_sparse_lidar_minmax_min_apollo/results/waymo_226_1008/orig_pred/226__images__000000_0.npz', 
    ) -> None:
    """Test entry1"""
    
    # read depth
    ixt = np.asarray([[2056.28, 0., 939.57], [0., 2056.28, 641.10], [0., 0., 1.]])
    orig_h, orig_w = 1280, 1920
    
    depth = np.load(depth_path)['data']
    h, w = depth.shape
    ixt_depth = ixt.copy()
    ixt_depth[:1] = ixt_depth[:1] * w / orig_w
    ixt_depth[1:2] = ixt_depth[1:2] * h / orig_h
    # depth[:240] = 0.
    depth_pcd = depth2pcd(depth, ixt_depth, depth_min=1., depth_max=50.)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(depth_pcd)
    o3d.io.write_point_cloud('test_apollo.ply', pcd)
    
    lidar_info = np.load(lidar_path, allow_pickle=True).item()
    lidar_mask, lidar_value = lidar_info['mask'], lidar_info['value']
    lidar_depth = np.zeros_like(lidar_mask).astype(np.float32)
    lidar_depth[lidar_mask] = lidar_value
    # import ipdb; ipdb.set_trace()
    lidar_pcd = depth2pcd(lidar_depth, ixt, depth_min=1., depth_max=50.)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_pcd)
    o3d.io.write_point_cloud('test_lidar.ply', pcd)
    

    
    import ipdb; ipdb.set_trace()
    
    pass
    
def entry2() -> None:
    """Test entry2"""
    pass
    
if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict(
        {
            "validate_align": validate_align,
            "entry2": entry2,
        }
    )


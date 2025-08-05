from lib.model.depth_estimation.depth_anything.depth_anything_upsample_fusion import DepthAnythingPipeline
import torch.nn.functional as F
import torch
import numpy as np
import cv2
import open3d as o3d
def generate_point_cloud(rgb_image, depth_image, camera_matrix,return_color=False):
    # import pdb;pdb.set_trace()
    v, u = depth_image.nonzero() 
    z = depth_image[v,u]
    x = (u - camera_matrix[0, 2]) * z / camera_matrix[0, 0]
    y = (v - camera_matrix[1, 2]) * z / camera_matrix[1, 1]
    point_cloud = np.stack((x, y, z), axis=-1)
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(point_cloud)
    pcd.colors=o3d.utility.Vector3dVector(rgb_image[v,u])
    return pcd


import torch
import numpy as np
from enum import Enum, auto

from scipy.spatial import KDTree

def interp_depth_rgb(sdpt: np.ndarray, 
                     ref_img: np.ndarray, 
                     speed=1, 
                     k=4,
                     **kwargs):
    h, w  = sdpt.shape
    lb = 0.
    if (sdpt <= lb).all(): return np.ones((h, w)) * lb, np.zeros((h, w))
    
    val_x, val_y = np.where(sdpt > lb)
    inval_x, inval_y = np.where(sdpt <= lb)
    val_pos = np.stack([val_x, val_y], axis=1)
    inval_pos = np.stack([inval_x, inval_y], axis=1)
    ref_img_min = ref_img.min()
    ref_img_max = ref_img.max()
    ref_img = (ref_img - ref_img_min) / ((ref_img_max - ref_img_min) + 0.0001)
    val_color = ref_img[val_x, val_y]
    inval_color = ref_img[inval_x, inval_y]

    if (sdpt != 0).sum() < k:
        k = (sdpt != 0).sum()

    tree = KDTree(val_pos)
    dists, inds = tree.query(inval_pos, k=k)

    try:
        dpt = np.copy(sdpt).reshape(-1)
    except:
        #print(kwargs["frame"])
        import ipdb; ipdb.set_trace()
    ######################
    if dpt.size == 0: import ipdb; ipdb.set_trace()
    ######################
    if k == 1:
        dpt[inval_x * w + inval_y] = sdpt.reshape(-1,)[val_pos[inds][..., 0] * w + val_pos[inds][..., 1]]
    else:
        ######################
        dists = np.where(dists == 0, 1e-10, dists)
        weights = 1 / dists
        weights /= np.sum(weights, axis=1, keepdims=True)

        rgb_diff = (inval_color[:, None] - val_color[inds])
        rgb_sim = -rgb_diff * 0.5 + 0.5
        rgb_sim = np.exp(speed*rgb_sim) * 0.01
        rgb_weights = rgb_sim / np.sum(rgb_sim, axis=1, keepdims=True)

        weights = weights * rgb_weights
        # weights = weights
        weights = weights / np.clip(np.sum(weights, axis=1, keepdims=True), 0.01, None)
        try:
            dpt = np.copy(sdpt).reshape(-1)
        except:
            import ipdb; ipdb.set_trace()

        try:
            nearest_vals = sdpt[val_x[inds], val_y[inds]]
        except:
            print(sdpt.shape, val_x.shape, val_y.shape, inds.shape, weights.shape, dists.shape, (sdpt != 0).sum())
        
        weighted_avg = np.sum(nearest_vals * weights, axis=1)
        dpt[inval_x * w + inval_y] = weighted_avg

    return dpt.reshape(h, w)

def project_to_image_depth(pts_3d, P):
    """ Project 3d points to image plane.

    Usage: pts_2d = projectToImage(pts_3d, P)
      input: pts_3d: nx3 matrix
             P:      3x4 projection matrix
      output: pts_2d: nx2 matrix

      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
      => normalize projected_pts_2d(2xn)

      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
    """
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    # print(('pts_3d_extend shape: ', pts_3d_extend.shape))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    u = pts_2d[:, 0] / pts_2d[:, 2]
    v = pts_2d[:, 1] / pts_2d[:, 2]
    depth = pts_2d[:, 2]
    return u, v, depth

def proj_pcd2frame_getdepth(P, pts, colorImage):
    """
    将点云投影到相机坐标系下，并获取深度图像
    
    Args:
        v_data (object): 包含相机位姿等信息的对象
        camera (object): 相机对象
        frame (int): 当前帧的索引
        pcd (object): 点云对象
    
    Returns:
        np.ndarray: 深度图像，形状为(height, width, 1)
    
    """
    height = colorImage.shape[0]
    width = colorImage.shape[1]



    u, v, depth = project_to_image_depth(pts, P)
    u = u.astype(np.int32)
    v = v.astype(np.int32)
    mask = np.logical_and(np.logical_and(u >= 0, u < width),
                          np.logical_and(v >= 0, v < height))
    mask = np.logical_and(mask, depth > 0)
    depthImage = np.zeros((height, width, 1))


    depthImage[v[mask], u[mask]] = depth[mask].reshape(-1, 1)
    return depthImage

# depth_model  =DepthAnythingPipeline(load_pretrain_net='checkpoints/e099-s200000.ckpt').cuda()
depth_model  =DepthAnythingPipeline(load_pretrain_net='checkpoints/v2_model_metric_ft_shift_minmax_fov.ckpt').cuda()
depth_model=depth_model.eval()

# import pdb;pdb.set_trace()
#  u, v, depth = project_to_image_depth(pts, P)
import time 
total_time=0

for _ in range(20):
    start=time.time()
    # pts=np.array(o3d.io.read_point_cloud('0000_0001_0000.ply').points)
    K=np.load('K.npy')
    image=cv2.imread('color.png')
    depth_map=np.load('lidar.npy')/100
    lowres_depth=depth_map

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lowres_depth = interp_depth_rgb(
        lowres_depth,
        gray,
        speed=5,
        k = 4
    )
    lowres_depth = cv2.resize(lowres_depth, (518,294), interpolation=cv2.INTER_LINEAR)
    image=cv2.resize(image/255, (518,294), interpolation=cv2.INTER_LINEAR)


    lowres_depth=torch.tensor(lowres_depth)[None,None]
    image=torch.tensor(image)[None].permute(0,3,1,2)

    with torch.inference_mode(),torch.no_grad():
        batch={'lowres_depth':lowres_depth.to(torch.float32).cuda(),'image':image.to(torch.float32).cuda()}
        result=depth_model.forward_test(batch)
        depth=result['depth']


    total_time+=time.time()-start

print(total_time/20)
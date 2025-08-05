import numpy as np
import open3d as o3d

def depth2pcd(depth, ixt, depth_min=None, depth_max=None, color=None, ext=None, conf=None):
    height, width = depth.shape
    x = np.arange(0, width)
    y = np.arange(0, height)
    xx, yy = np.meshgrid(x, y)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    zz = depth.reshape(-1)
    mask = np.ones_like(xx, dtype=np.bool_)
    if depth_min is not None:
        mask &= zz >= depth_min
    if depth_max is not None:
        mask &= zz <= depth_max
    if conf is not None:
        mask &= conf.reshape(-1) == 2
    xx = xx[mask]
    yy = yy[mask]
    zz = zz[mask]
    pcd = np.stack([xx, yy, np.ones_like(xx)], axis=1)
    pcd = pcd * zz[:, None]
    pcd = np.dot(pcd, np.linalg.inv(ixt).T)
    if ext is not None:
        pcd = np.concatenate([pcd, np.ones((pcd.shape[0], 1))], axis=1)
        pcd = np.dot(pcd, np.linalg.inv(ext).T)
    if color is not None:
        return pcd[:, :3], color.reshape(-1, 3)[mask]
    else:
        return pcd[:, :3]
    
def export_pcd(path, points, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud(path, pcd)
    
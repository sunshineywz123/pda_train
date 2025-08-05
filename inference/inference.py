from lib.model.depth_estimation.depth_anything.depth_anything_upsample_fusion import DepthAnythingPipeline
import torch.nn.functional as F
import torch
import numpy as np
import cv2
import open3d as o3d
def generate_point_cloud(rgb_image, depth_image, camera_matrix,return_color=False):

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


# depth_model  =DepthAnythingPipeline(load_pretrain_net='checkpoints/e099-s200000.ckpt').cuda()
depth_model  =DepthAnythingPipeline(load_pretrain_net='checkpoints/v2_model_metric_ft_shift_minmax_fov.ckpt').cuda()

depth_model=depth_model.eval()
x=np.load('test/000000_0.npy',allow_pickle=True).item()
depth_map=x['mask'].astype(np.float32)
depth_map[depth_map!=0]=x['value']

lowres_depth=depth_map
image=cv2.imread('test/000000_0.png')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# lowres_depth = interp_depth_rgb(
#     lowres_depth,
#     gray,
#     speed=5,
#     k = 4
# )
lowres_depth = cv2.resize(lowres_depth, (756, 504), interpolation=cv2.INTER_LINEAR)
image=cv2.resize(image/255,(756, 504), interpolation=cv2.INTER_LINEAR)


# import pdb;pdb.set_trace()
lowres_depth=torch.tensor(lowres_depth)[None,None]
image=torch.tensor(image)[None].permute(0,3,1,2)

# lowres_depth = F.interpolate(lowres_depth, (504,756))
# image = F.interpolate(image, (504,756))


with torch.inference_mode(),torch.no_grad():
    batch={'lowres_depth':lowres_depth.to(torch.float32).cuda(),'image':image.to(torch.float32).cuda()}
    result=depth_model.forward_test(batch)
    depth=result['depth']

image=image.permute(0,2,3,1)[0].detach().cpu().numpy()
depth=depth[0][0].detach().cpu().numpy()
lowres_depth=lowres_depth[0][0].detach().cpu().numpy()
coff=1920/756
K=np.array([
    [2015/coff,0,1920/(2*coff)],
    [0,2015/coff,1280/(2*coff)],
    [0,0,1],
])
o3d.io.write_point_cloud('input.ply',generate_point_cloud(image,lowres_depth,K).uniform_down_sample(10))
o3d.io.write_point_cloud('predict.ply',generate_point_cloud(image,depth,K).uniform_down_sample(10))
o3d.io.write_point_cloud('gt.ply',generate_point_cloud(image,np.load('test/226__images__000000_0.npz',allow_pickle=True)['data'],K).uniform_down_sample(10))
# import pdb;pdb.set_trace()
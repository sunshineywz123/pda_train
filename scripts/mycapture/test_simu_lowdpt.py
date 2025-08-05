import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
import imageio
sys.path.append('.')
import cv2
from lib.utils import vis_utils
import matplotlib.pyplot as plt
img_h, img_w = 192, 256
# img_h, img_w = 384, 512
from scipy.spatial import KDTree
import torch
import torch.nn.functional as F



def bilateralFilter(src, d, sigmaColor, sigmaSpace, depth):
    '''
    INPUT:
    src: input image
    d: 	Diameter of each pixel neighborhood that is used during filtering. 
    sigmaColor: Filter sigma in the color space
    sigmaSpace: Filter sigma in the coordinate space.
    
    OUTPUT:
    dst: return image 
    '''
    print("Running Bilateral Blur")
    assert src.dtype == np.uint8
    src = src.astype(np.int64) # avoid overflow
    ksize = (d, d)
    ret = np.zeros_like(depth)
    H, W = src.shape[:2]
    assert(ksize[0] % 2 == 1 and ksize[0] > 0)
    assert(ksize[1] % 2 == 1 and ksize[1] > 0)
    offsetX, offsetY = np.meshgrid(np.arange(ksize[0]), np.arange(ksize[1]))
    offsetX -= ksize[0] // 2
    offsetY -= ksize[1] // 2
    w1 = np.exp(-(offsetX ** 2 + offsetY ** 2) / (2 * sigmaSpace ** 2))
    from tqdm import tqdm
    for i in tqdm(range(0, H)):
        for j in range(0, W):
            indY = offsetY + i
            indX = offsetX + j
            indY[indY < 0] = 0
            indY[indY >= H] = H - 1
            indX[indX < 0] = 0
            indX[indX >= W] = W - 1
            cropped_img = src[indY, indX]
            cropped_depth = depth[indY, indX]
            diff = -(np.sum((cropped_img - src[i, j]) ** 2, axis = -1) / (2 * sigmaColor ** 2) )
#             w2 = np.exp(diff / (2 * sigmaColor ** 2))
            w2 = np.exp(diff)
            ret[i, j] = (w1 * w2 * cropped_depth).reshape(-1).sum() / (w1 * w2).sum()
#             break
#         break
    return ret

def GenerateSpotMask(stride=11, dist_coef=2e-5, noise=0, plt_flag=False):
    '''
    Simulate pincushion distortion:
    --stride: 
    It controls the distance between neighbor spots7
    Suggest stride value:       5/6

    --dist_coef:
    It controls the curvature of the spot pattern
    Larger dist_coef distorts the pattern more.
    Suggest dist_coef value:    0 ~ 5e-5

    --noise:
    standard deviation of the spot shift
    Suggest noise value:        0 ~ 0.5
    '''

    # Generate Grid points
    hline_num = img_h//stride
    x_odd, y_odd = np.meshgrid(np.arange(stride//2, img_h, stride*2), np.arange(stride//2, img_w, stride))
    x_even, y_even = np.meshgrid(np.arange(stride//2+stride, img_h, stride*2), np.arange(stride, img_w, stride))
    x_u = np.concatenate((x_odd.ravel(),x_even.ravel()))
    y_u = np.concatenate((y_odd.ravel(),y_even.ravel()))
    x_u -= img_h//2
    y_u -= img_w//2

    # Distortion
    r_u = np.sqrt(x_u**2+y_u**2)
    r_d = r_u + dist_coef * r_u**3
    num_d = r_d.size
    sin_theta = x_u/r_u
    cos_theta = y_u/r_u
    x_d = np.round(r_d * sin_theta + img_h//2 + np.random.normal(0, noise, num_d))
    y_d = np.round(r_d * cos_theta + img_w//2 + np.random.normal(0, noise, num_d))
    idx_mask = (x_d<img_h) & (x_d>0) & (y_d<img_w) & (y_d>0)
    x_d = x_d[idx_mask].astype('int')
    y_d = y_d[idx_mask].astype('int')

    spot_mask = np.zeros((img_h, img_w))
    spot_mask[x_d,y_d] = 1

    # Plot mask
    if plt_flag:
        plt.scatter(y_d,x_d, np.ones_like(x_d))
        plt.xlim([0,img_w])
        plt.ylim([0,img_h])
        # plt.title(f'Spot mask (spot number: {y_d.size})')
        # plt.show()

    return spot_mask, x_d, y_d

def interp_depth(sdpt: np.ndarray, dist_func="euclidean_norm", **kwargs):
    ######################
    if sdpt.size == 0: import ipdb; ipdb.set_trace()
    ######################
    h, w  = sdpt.shape

    # lb = 0 if (sdpt >= 0.).all() else -1.
    lb = 0.
    if (sdpt <= lb).all(): return np.ones((h, w)) * lb, np.zeros((h, w))

    # interpolation
    val_x, val_y = np.where(sdpt > lb)
    inval_x, inval_y = np.where(sdpt <= lb)
    val_pos = np.stack([val_x, val_y], axis=1)
    inval_pos = np.stack([inval_x, inval_y], axis=1)

    k = kwargs.get("k", 1)
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

def interp_depth_rgb(sdpt: np.ndarray, rgb: np.ndarray, speed=1, dist_func="euclidean_norm", **kwargs):
    ######################
    if sdpt.size == 0: import ipdb; ipdb.set_trace()
    ######################
    h, w  = sdpt.shape

    # lb = 0 if (sdpt >= 0.).all() else -1.
    lb = 0.
    if (sdpt <= lb).all(): return np.ones((h, w)) * lb, np.zeros((h, w))

    # interpolation
    val_x, val_y = np.where(sdpt > lb)
    inval_x, inval_y = np.where(sdpt <= lb)
    val_pos = np.stack([val_x, val_y], axis=1)
    inval_pos = np.stack([inval_x, inval_y], axis=1)
    val_color = rgb[val_x, val_y] / 255.
    inval_color = rgb[inval_x, inval_y] / 255.

    k = kwargs.get("k", 1)
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
        rgb_diff = np.linalg.norm(rgb_diff, axis=-1)
        rgb_sim = np.sqrt(rgb.shape[-1]) + 0.001 - rgb_diff
        rgb_sim = np.exp(speed*rgb_sim)
        rgb_weights = rgb_sim / np.sum(rgb_sim, axis=1, keepdims=True)

        weights = weights * rgb_weights
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

def main(args):
    rgb_dir = '/mnt/data/home/linhaotong/datasets/mycapture/671d34a318/rgb'
    depth_dir = '/mnt/data/home/linhaotong/datasets/mycapture/671d34a318/depth'
    rgb_path = os.path.join(rgb_dir, '000000.jpg')
    depth_path = os.path.join(depth_dir, '000000.png')

    rgb_path = '/home/linhaotong/frame.0001.tonemap.jpg'
    depth_path = '/home/linhaotong/frame.0001.depth_meters.hdf5'
    import h5py
    depth = np.asarray(h5py.File(depth_path)['dataset']).astype(np.float32)
    depth = cv2.resize(depth, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    rgb = imageio.imread(rgb_path)
    rgb_h, rgb_w = rgb.shape[:2]
    encoder = 'vits'
    dinov2 = pretrained = torch.hub.load('{}/cache_models/depth_anything/torchhub/facebookresearch_dinov2_main'.format(os.environ['workspace']), 'dinov2_{:}14'.format(encoder), source='local', pretrained=True)
    dinov2.cuda()
    x = (torch.from_numpy(rgb).cuda().permute(2, 0, 1).unsqueeze(0).float() / 255.).float()
    x = (x - torch.tensor([0.485, 0.456, 0.406], device='cuda', dtype=torch.float32).view(1, 3, 1, 1) ) / torch.tensor([0.229, 0.224, 0.225], device='cuda', dtype=torch.float32).view(1, 3, 1, 1)
    input_h, input_w = rgb_h // 14 * 14, rgb_w // 14 * 14
    x = F.interpolate(x, size=(input_h, input_w), mode='bilinear', align_corners=False)
    features = dinov2.get_intermediate_layers(x, [2, 5, 8, 11], return_class_token=True)
    low_dim = 3
    feature_map = torch.pca_lowrank(features[-1][0], low_dim)[0].reshape(1, rgb_h//14, rgb_w//14, low_dim)
    feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
    feature_map = F.interpolate(feature_map.permute(0, 3, 1, 2), size=(img_h, img_w), mode='bilinear', align_corners=False)
    # feature_map = torch.pca_lowrank(feature_map, q=3)
    # import ipdb; ipdb.set_trace()
    # depth = np.asarray(imageio.imread(depth_path))/1000
    rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)
    # plt.subplot(131)
    spot = GenerateSpotMask(stride=7, dist_coef=1e-5, noise=0.1, plt_flag=False)
    spot_mask = spot[0] == 1.
    print(spot_mask.sum())
    sparse_depth = np.zeros_like(depth)
    sparse_depth[spot_mask] = depth[spot_mask]
    sparse_interp_depth_spatial = interp_depth(sparse_depth, k=4)
    # sparse_interp_depth = interp_depth_rgb(sparse_depth, rgb,  k=5, speed=5.)
    sparse_interp_depth_rgb = interp_depth_rgb(sparse_depth, 
                                        #    np.concatenate([cv2.resize(feature_map[0].detach().cpu().numpy(), (img_w, img_h), interpolation=cv2.INTER_NEAREST), rgb], axis=-1),  
                                        np.asarray((depth / depth.max())*255.).astype(np.uint8)[..., None].repeat(3, -1),
                                           k=4, 
                                           speed=8.)
    # sparse_interp_depth_bi = bilateralFilter(rgb, 9, sigmaColor=1, sigmaSpace=25., depth=sparse_interp_depth_rgb)
    sparse_interp_depth_bi = bilateralFilter(np.asarray((depth / depth.max())*255.).astype(np.uint8)[..., None].repeat(3, -1), 9, sigmaColor=1, sigmaSpace=25., depth=sparse_interp_depth_rgb)
    
    sparse_interp_depth_bi_resize = cv2.resize(sparse_interp_depth_bi, (256, 192), interpolation=cv2.INTER_NEAREST)

    plt.subplot(231)
    plt.imshow(rgb)
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(spot_mask)
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(cv2.resize(depth, (256, 192), interpolation=cv2.INTER_LINEAR))
    plt.axis('off')
    plt.subplot(234)
    plt.imshow(sparse_interp_depth_spatial)
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(sparse_interp_depth_rgb)
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(sparse_interp_depth_bi)
    plt.axis('off')
    plt.savefig('test.jpg', bbox_inches='tight', pad_inches=0, dpi=300)
    # rgb_dir = '/mnt/data/home/linhaotong/datasets/mycapture/671d34a318/rgb'
    # depth_dir = '/mnt/data/home/linhaotong/datasets/mycapture/671d34a318/depth'
    # conf_dir = '/mnt/data/home/linhaotong/datasets/mycapture/671d34a318/confidence'
    # tar_dir = 'data/pl_htcode/exp_results/video_depth_results/songyou_iphone_conf_depth'
    # read_depth_func = lambda x: np.asarray(imageio.imread(x))/1000
    # video = vis_utils.warp_rgbd_video(rgb_dir, depth_dir, tar_dir, conf_dir, read_depth_func=read_depth_func, frame_sample=[0, 30000, 2])

    # rgb_dir = '/mnt/data/home/linhaotong/datasets/mycapture/671d34a318/rgb'
    # depth_dir = '/mnt/data/home/linhaotong/projects/pl_htcode/data/pl_htcode/outputs/depth_estimation/july_hypersim_baseline0717_minmax_new_fixbug_1nodes/results/default/orig_pred'
    # tar_dir = 'data/pl_htcode/exp_results/video_depth_results/songyou_iphone_depth_minmax'
    # video = vis_utils.warp_rgbd_video(rgb_dir, depth_dir, tar_dir, frame_sample=[0, 6000, 2], depth_format='.npz')
    
    # rgb_dir = '/mnt/data/home/linhaotong/datasets/mycapture/671d34a318/rgb'
    # depth_dir = '/mnt/data/home/linhaotong/projects/pl_htcode/data/pl_htcode/outputs/depth_estimation/repeat_aug_zipmesh_arkit/results/default/orig_pred'
    # tar_dir = 'data/pl_htcode/exp_results/video_depth_results/songyou_iphone_depth_aug_zipmesh_arkit'
    # video = vis_utils.warp_rgbd_video(rgb_dir, depth_dir, tar_dir, frame_sample=[0, 6000, 2], depth_format='.npz')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)
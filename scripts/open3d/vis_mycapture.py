import os
from os.path import join
import argparse
import sys
# from arrow import get
import numpy as np
from tqdm import tqdm
sys.path.append('.')
import imageio
import open3d as o3d
from trdparties.colmap.read_write_model import read_model
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_root', type=str, default='data/pl_htcode/3d_scanner/processed/2024_04_16_16_13_19')
    parser.add_argument('--input_pred_root', type=str, default='data/pl_htcode/outputs/mde/test4nodes/results/mycapture_1920/mycapture_1920')
    parser.add_argument('--frame_id', type=int, default=354)
    args = parser.parse_args()
    return args

def unproject(rgb, dpt, ixt, color='rgb'):
    h, w = rgb.shape[:2]
    x, y = np.arange(w), np.arange(h)
    xx, yy = np.meshgrid(x, y)
    xyz = np.stack((xx, yy, np.ones_like(xx)), axis=-1).astype(np.float32)
    xyz *= dpt[..., None]
    point_cloud = xyz @ np.linalg.inv(ixt.T)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud.reshape(-1, 3))
    rgb = rgb.copy()
    if color == 'red': rgb[..., :1] = 1.
    elif color == 'green': rgb[..., 1:2] = 1.
    elif color == 'blue': rgb[..., 2:3] = 1.
    elif color == 'rgb': pass
    pcd.colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3))
    return pcd

def read_depth(args):
    if args.input_dpt_path[-3:] == 'npz':
        # hypersim gt depth
        dpt = np.asarray(np.load(args.input_dpt_path)['data'])
    elif args.input_dpt_path[-3:] in ['npy', 'npz']:
        import ipdb; ipdb.set_trace()
    else:
        dpt = imageio.imread(args.input_dpt_path)
        if dpt.dtype == np.uint8: dpt = dpt.astype(np.float32) / 255.
    print(dpt.min(), dpt.max())
    if len(dpt.shape) == 3: dpt = dpt[..., 0]
    if args.depth_type == 'log':
        dpt = np.exp(dpt * (np.log(80.) - np.log(0.5)) + np.log(0.5))
    elif args.depth_type == 'disp':
        dpt = np.clip(dpt, 1/80., None)
        dpt = 1/dpt
    else:
        pass
    dpt = dpt * args.scale + args.shift
    min_val = np.percentile(dpt, args.filter_min)
    max_val = np.percentile(dpt, args.filter_max)
    msk = np.logical_and(dpt>min_val, dpt<max_val)
    return dpt, msk
    

def main(args):
    rgb = np.asarray(imageio.imread(join(args.input_data_root, 'rgb/frame_{:05d}.jpg'.format(args.frame_id))) / 255.).astype(np.float32)
    lowres_depth = (imageio.imread(join(args.input_data_root, 'depth/depth_{:05d}.png'.format(args.frame_id))) / 1000.)
    highres_depth = cv2.resize(lowres_depth, (1920, 1440), interpolation=cv2.INTER_LINEAR)
    lowres_rgb = cv2.resize(rgb, (256, 192), interpolation=cv2.INTER_AREA)
    pred = np.load(join(args.input_pred_root, 'orig_pred/frame_{:05d}.npz'.format(args.frame_id)))['data']
    
    cams, _, _ = read_model(join(args.input_data_root, 'colmap/sparse'))
    cam_id = 1
    ixt = np.asarray([[cams[cam_id].params[0], 0, cams[cam_id].params[2]], [0, cams[cam_id].params[1], cams[cam_id].params[3]], [0, 0, 1]])
    
    scale = 1920 / 256 
    ixt_lowres = ixt.copy()
    ixt_lowres[:2] /= scale
    
    
    # lowres_depth_pcd = unproject(lowres_rgb, lowres_depth, ixt_lowres) 
    # highres_depth_pcd = unproject(rgb, highres_depth, ixt)
    # pred_pcd = unproject(rgb, pred, ixt)
    # o3d.visualization.draw_geometries([lowres_depth_pcd, highres_depth_pcd, pred_pcd])
    
    import ipdb; ipdb.set_trace()
    start_x, start_y = 800, 400
    end_x, end_y = 1200, 800
    
    start_x_lowres, start_y_lowres = int(start_x / scale), int(start_y / scale)
    end_x_lowres, end_y_lowres = int(end_x / scale), int(end_y / scale)
    
    import matplotlib.pyplot as plt 
    plt.subplot(221)
    plt.imshow(rgb[start_y:end_y, start_x:end_x])
    plt.subplot(222)
    import ipdb; ipdb.set_trace()
    plt.imshow(lowres_depth[start_y_lowres:end_y_lowres, start_x_lowres:end_x_lowres])
    plt.subplot(223)
    plt.imshow(pred[start_y:end_y, start_x:end_x])
    plt.subplot(224)
    plt.imshow(highres_depth[start_y:end_y, start_x:end_x])
    plt.axis('off')
    plt.show()
    
    
if __name__ == '__main__':
    args = parse_args()
    main(args)

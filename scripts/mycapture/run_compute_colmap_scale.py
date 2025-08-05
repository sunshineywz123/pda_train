import os
from os.path import join
import argparse
import sys
import ipdb
import numpy as np
from tqdm import tqdm
import imageio
import json

sys.path.append('.')
from lib.utils.pylogger import Log
from scipy.optimize import root
import cv2

from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from trdparties.colmap.read_write_model import Image, Point3D, read_cameras_binary, read_images_binary, read_model, write_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/mnt/bn/haotongdata/Datasets/mycapture_stray_scanner')
    parser.add_argument('--scene', type=str, default='04042880d2')
    args = parser.parse_args()
    return args

def compute_scale(colmap_path, depth_path, conf_path=None, thresh_views=15):
    '''
    return scale from colmap to metric depth
    '''
    cams, images, points3d = read_model(colmap_path)
    name2id = {image.name:k for k, image in images.items()}
    names = [image.name for k, image in images.items()]
    names = sorted(names)

    colmap_depth, metric_depth = [], []
    for idx, name in tqdm(enumerate(names)):
        if 'for_' in  name:
            continue
        image = images[name2id[name]]
        ext = np.eye(4)
        ext[:3, :3] = image.qvec2rotmat()
        ext[:3, 3] = image.tvec
        ixt = np.eye(3)
        cam_id = image.camera_id
        camera = cams[cam_id]
        cam_height, cam_width = camera.height, camera.width
        ixt[0, 0], ixt[1, 1], ixt[0, 2], ixt[1, 2] = camera.params[:4]
        xys = image.xys[image.point3D_ids != -1]
        views = np.asarray([len(points3d[id].image_ids) for id in image.point3D_ids if id != -1])
        points = np.asarray([points3d[id].xyz for id in image.point3D_ids if id != -1])
        sparse_depth = (points @ ext[:3, :3].T + ext[:3, 3:].T)[..., 2]
        uv = xys 

        if (views > thresh_views).sum() == 10:
            continue

        uv_msk = views > thresh_views
        if uv_msk.sum() == 0:
            continue
        depth = (cv2.imread(join(depth_path,name.replace('.jpg', '.png')), cv2.IMREAD_ANYDEPTH) / 1000.).astype(np.float32)
        uv[..., 0] *= (depth.shape[1] / cam_width)
        uv[..., 1] *= (depth.shape[0] / cam_height)
        uv = uv.astype(np.int32)
        uv[..., 0] = np.clip(uv[..., 0], 0, depth.shape[1] - 1)
        uv[..., 1] = np.clip(uv[..., 1], 0, depth.shape[0] - 1)
        metric_depth_item = depth[uv[:, 1], uv[:, 0]]
        uv_msk = uv_msk & (metric_depth_item < 4.)
        if conf_path is not None:
            if os.path.exists(join(conf_path, name.replace('.jpg', '.png'))):
                conf = cv2.imread(join(conf_path, name.replace('.jpg', '.png')), cv2.IMREAD_ANYDEPTH)
                confidence = conf[uv[:, 1], uv[:, 0]]
                uv_msk = uv_msk & (confidence == 2)
        if uv_msk.sum() > 10:
            colmap_depth.append(sparse_depth[uv_msk])
            metric_depth.append(metric_depth_item[uv_msk])
    colmap_depth = np.concatenate(colmap_depth, axis=0)
    metric_depth = np.concatenate(metric_depth, axis=0)

    degree = 1
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_model = LinearRegression(fit_intercept=False)
    ransac = RANSACRegressor(estimator=linear_model, max_trials=100000)
    model = make_pipeline(poly_features, ransac)
    model.fit(colmap_depth[:, None], metric_depth[:, None])
    a, b = model.named_steps['ransacregressor'].estimator_.coef_, model.named_steps['ransacregressor'].estimator_.intercept_
    a = a.item()
    try: b = b.item()
    except: pass
    Log.info(f'colmap2metric: {a} {b}')
    ## DEBUG
    # import matplotlib.pyplot as plt
    # rand_msk = np.random.choice(colmap_depth.shape[0], 1000)
    # rand_colmap_depth = colmap_depth[rand_msk]
    # rand_metric_depth = metric_depth[rand_msk]
    # # sorting rand metric depth
    # sort_idx = np.argsort(rand_metric_depth)
    # rand_metric_depth = rand_metric_depth[sort_idx]
    # rand_colmap_depth = rand_colmap_depth[sort_idx]
    # plt.plot(np.arange(rand_metric_depth.shape[0]), rand_metric_depth, label='metric', linewidth=2.)
    # plt.plot(np.arange(rand_colmap_depth.shape[0]), rand_colmap_depth, label='colmap', linewidth=1, linestyle='--')
    # plt.plot(np.arange(rand_colmap_depth.shape[0]), a * rand_colmap_depth + b, label='regression', linewidth=0.5, linestyle='-.')
    # plt.savefig('depth.png')
    return a
    
def main(args):
    colmap_path = join(args.input_path, args.scene, 'colmap/sparse/0')
    os.makedirs(colmap_path, exist_ok=True)
    os.system('mv {}/*.bin {}/0'.format(join(args.input_path, args.scene, 'colmap/sparse'), join(args.input_path, args.scene, 'colmap/sparse')))
    cmd = 'colmap model_orientation_aligner --input_path {} --output_path {} --image_path {}'.format(
        join(args.input_path, args.scene, 'colmap/sparse/0'),
        join(args.input_path, args.scene, 'colmap/sparse/0'),
        join(args.input_path, args.scene, 'images')
    )
    os.system(cmd)
    lidar_depth_path = join(args.input_path, args.scene, 'depth')
    conf_depth_path = join(args.input_path, args.scene, 'confidence')
    if os.path.exists(conf_depth_path): conf_depth_path = None
    scale = compute_scale(colmap_path, lidar_depth_path, conf_path=conf_depth_path)
    cameras, images, points3d = read_model(colmap_path)
    new_images = {}
    for k in images:
        image = images[k]
        new_image = Image(image.id, image.qvec, image.tvec * scale, image.camera_id, image.name, image.xys, image.point3D_ids)
        new_images[k] = new_image
    new_points3d = {}
    for k in points3d:
        point3d = points3d[k]
        new_point3d = Point3D(point3d.id, point3d.xyz * scale, point3d.rgb, point3d.error, point3d.image_ids, point3d.point2D_idxs)
        new_points3d[k] = new_point3d
    os.makedirs(join(args.input_path, args.scene, 'colmap/sparse/0_metric'), exist_ok=True)
    write_model(cameras, new_images, new_points3d, join(args.input_path, args.scene, 'colmap/sparse/0_metric'))
    Log.info('Write metric colmap model to ' + join(args.input_path, args.scene, 'colmap/sparse/0_metric'))
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
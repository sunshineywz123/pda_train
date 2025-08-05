import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
import imageio



from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from PIL import Image as PILImage
from lib.utils.pylogger import Log
from trdparties.colmap.read_write_model import Image, Point3D, read_model, write_model
sys.path.append('.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/bn/haotongdata/Datasets/mycapture_stray_scanner/calib5')
    parser.add_argument('--colmap_path', type=str, default='colmap/sparse_rp/0_align')
    parser.add_argument('--output_colmap_path', type=str, default='colmap/sparse_rp/0_metric')
    parser.add_argument('--depth_conf_path', type=str, default='background')
    args = parser.parse_args()
    return args

im2cam = {
    'lidar_cam01/lidar_cam01.jpg': '30',
    'lidar_cam02/lidar_cam02.jpg': '31',
    'lidar_cam03/lidar_cam03.jpg': '32',
    'lidar_cam04/lidar_cam04.jpg': '33',
}

def main(args):
    colmap_path = join(args.input, args.colmap_path)
    output_colmap_path = join(args.input, args.output_colmap_path)
    os.makedirs(output_colmap_path, exist_ok=True)
    
    cameras, images, points3D = read_model(colmap_path)

    sparse_depths = []
    metric_depths = []

    imgs = []

    for im_id in tqdm(images):
        if 'lidar_cam' not in images[im_id].name:
            continue
        cam_id = im2cam[images[im_id].name]

        ext = np.eye(4)
        ext[:3, :3] = images[im_id].qvec2rotmat()
        ext[:3, 3] = images[im_id].tvec
        
        mini_thresh_views = 8
        views = np.asarray([len(points3D[point_id].image_ids) for point_id in images[im_id].point3D_ids if point_id != -1])
        points = np.asarray([points3D[point_id].xyz for point_id in images[im_id].point3D_ids if point_id != -1])
        xys = np.asarray(images[im_id].xys[images[im_id].point3D_ids != -1])
        points = points @ ext[:3, :3].T + ext[:3, 3]
        sparse_depth = points[:, 2]

        cam_height, cam_width = cameras[images[im_id].camera_id].height, cameras[images[im_id].camera_id].width
        depth = imageio.imread(join(args.input, args.depth_conf_path, '{}_depth.png'.format(cam_id)))
        if 'cam01' in images[im_id].name or 'cam02' in images[im_id].name:
            depth_image_pil = PILImage.fromarray(depth)
            depth = depth_image_pil.rotate(180)
            depth = np.asarray(depth)
        depth_height, depth_width = depth.shape
        depth = depth / 1000
        imgs.append(np.copy(depth))
        uv = np.stack([xys[:, 0] * depth_width / cam_width, xys[:, 1] * depth_height / cam_height], axis=-1)
        depth = depth[uv[:, 1].astype(int), uv[:, 0].astype(int)]
        conf = imageio.imread(join(args.input, args.depth_conf_path, '{}_conf.png'.format(cam_id)))
        if 'cam01' in images[im_id].name or 'cam02' in images[im_id].name:
            conf_image_pil = PILImage.fromarray(conf)
            conf = conf_image_pil.rotate(180)
            conf = np.asarray(conf)
        conf = conf[uv[:, 1].astype(int), uv[:, 0].astype(int)]

        msk = (conf == 2) & (views > mini_thresh_views) & (depth < 5)
        sparse_depths.extend(sparse_depth[msk])
        metric_depths.extend(depth[msk])

    import matplotlib.pyplot as plt
    plt.imshow(np.concatenate(imgs, axis=1))
    plt.savefig('test.jpg')
    plt.close()
    sparse_depth = np.asarray(sparse_depths)
    metric_depth = np.asarray(metric_depths)
    Log.info(f'sparse_depth: {sparse_depth.shape}')

    degree = 1
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_model = LinearRegression(fit_intercept=False)
    ransac = RANSACRegressor(estimator=linear_model, max_trials=100000)
    model = make_pipeline(poly_features, ransac)
    model.fit(sparse_depth[:, None], metric_depth[:, None])
    a, b = model.named_steps['ransacregressor'].estimator_.coef_, model.named_steps['ransacregressor'].estimator_.intercept_
    a = a.item()
    try: b = b.item()
    except: pass
    Log.info(f'colmap2metric: {a} {b}')

    scale = a
    new_images = {}
    for k in images:
        image = images[k]
        new_image = Image(image.id, image.qvec, image.tvec * scale, image.camera_id, image.name, image.xys, image.point3D_ids)
        new_images[k] = new_image
    
    points3d = points3D
    
    new_points3d = {}
    for k in points3d:
        point3d = points3d[k]
        new_point3d = Point3D(point3d.id, point3d.xyz * scale, point3d.rgb, point3d.error, point3d.image_ids, point3d.point2D_idxs)
        new_points3d[k] = new_point3d
    write_model(cameras, new_images, new_points3d, output_colmap_path)
    Log.info('Write metric colmap model to ' + output_colmap_path)
    

if __name__ == '__main__':
    args = parse_args()
    main(args)
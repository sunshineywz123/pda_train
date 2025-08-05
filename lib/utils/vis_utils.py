import numpy as np
import cv2
import matplotlib
import torch
import os
from os.path import join
import imageio
import open3d as o3d

from lib.utils import geo_utils
from lib.utils.parallel_utils import parallel_execution
from lib.utils.geo_utils import depth2pcd
from lib.utils.pylogger import Log
def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().clone().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored


row_col_ = {
    2: (2, 1),
    7: (2, 4),
    8: (2, 4),
    9: (3, 3),
    26: (4, 7)
}

row_col_square = {
    2: (2, 1),
    7: (3, 3),
    8: (3, 3),
    9: (3, 3),
    26: (5, 5)
}

def get_row_col(l, square):
    if square and l in row_col_square.keys():
        return row_col_square[l]
    if l in row_col_.keys():
        return row_col_[l]
    else:
        from math import sqrt
        row = int(sqrt(l) + 0.5)
        col = int(l/ row + 0.5)
        if row*col<l:
            col = col + 1
        if row > col:
            row, col = col, row
        return row, col

def merge(images, row=-1, col=-1, resize=False, ret_range=False, square=False, resize_height=1000, **kwargs):
    if row == -1 and col == -1:
        row, col = get_row_col(len(images), square)
    height = images[0].shape[0]
    width = images[0].shape[1]
    # special case
    if height > width:
        if len(images) == 3:
            row, col = 1, 3
    if len(images[0].shape) > 2:
        ret_img = np.zeros((height * row, width * col, images[0].shape[2]), dtype=np.uint8) + 255
    else:
        ret_img = np.zeros((height * row, width * col), dtype=np.uint8) + 255
    ranges = []
    for i in range(row):
        for j in range(col):
            if i*col + j >= len(images):
                break
            img = images[i * col + j]
            # resize the image size
            img = cv2.resize(img, (width, height))
            ret_img[height * i: height * (i+1), width * j: width * (j+1)] = img
            ranges.append((width*j, height*i, width*(j+1), height*(i+1)))
    if resize:
        min_height = resize_height
        if ret_img.shape[0] > min_height:
            scale = min_height/ret_img.shape[0]
            ret_img = cv2.resize(ret_img, None, fx=scale, fy=scale)
            ranges = [(int(r[0]*scale), int(r[1]*scale), int(r[2]*scale), int(r[3]*scale)) for r in ranges]
    if ret_range:
        return ret_img, ranges
    return ret_img

def warp_rgbd_video(img_dir, depth_dir, 
                    tar_dir=None,
                    conf_dir=None,
                    read_depth_func=None, 
                    frame_sample=[0, 600, 1], 
                    focal=1440, 
                    tar_h=756,
                    tar_w=1008,
                    cx=None, 
                    cy=None, 
                    depth_format='.png', 
                    depth_prefix=None,
                    depth_min=0.1,
                    depth_max=10.,
                    pre_upsample=False,
                    fps=60):
    # focal of iphone 13 pro: 1440-1443
    # focal of iphone 15 pro / Ipad: 1350-1400
    img_files = sorted([join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    depth_files = sorted([join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith(depth_format)])
    
    if (len(img_files) - len(depth_files)) >= 2 or (len(depth_files) - len(img_files) >= 2):
        depth_files = []
        for i in range(len(img_files)): depth_files.append(join(depth_dir, os.path.basename(img_files[i]).replace('.jpg', '.png'))) 
    if depth_prefix is not None:
        depth_files = [f for f in depth_files if os.path.basename(f).startswith(depth_prefix)]
    
    if conf_dir is not None:
        conf_files = sorted([join(conf_dir, f) for f in os.listdir(conf_dir) if f.endswith('.png')])
    else:
        conf_files = [None for _ in depth_files]

    if (len(img_files) - 1) == len(depth_files):
        img_files = img_files[:-1]
    elif len(img_files) == (len(depth_files) - 1):
        depth_files = depth_files[:-1]
        conf_files = conf_files[:-1]
    
    # assert len(img_files) == len(depth_files), 'The number of images and depth maps should be the same.'
    # assert len(img_files) == len(conf_files)
    frame_len = len(img_files)
    if frame_sample[1] == -1: frame_sample[1] = frame_len

    img_files = img_files[frame_sample[0]:frame_sample[1]:frame_sample[2]]
    depth_files = depth_files[frame_sample[0]:frame_sample[1]:frame_sample[2]]
    conf_files = conf_files[frame_sample[0]:frame_sample[1]:frame_sample[2]]
    if tar_dir is not None: tar_files = [join(tar_dir, 'rgb/{:06d}.jpg'.format(i)) for i in range(len(img_files))]
    else: tar_files = [None for img_file in img_files]
    fps = int(fps / frame_sample[2])


    # debug
    # img_files = img_files[:1]
    # warp_rgbd(img=img_files[0], depth=depth_files[0], tar_path=tar_files[0], conf=conf_files[0], focal=focal, tar_h=tar_h, tar_w=tar_w, cx=cx, cy=cy, depth_min=depth_min, depth_max=depth_max, read_depth_func=read_depth_func)
    import ipdb; ipdb.set_trace()
    
    parallel_execution(
        img_files,
        depth_files,
        conf_files,
        tar_files,
        focal,
        tar_h,
        tar_w,
        cx,
        cy,
        depth_min,
        depth_max,
        read_depth_func,
        pre_upsample,
        action=warp_rgbd,
        print_progress=True,
        desc="Warping RGBD",
    )
    if tar_dir is not None:
        if os.path.exists(join(tar_dir, 'rgb.mp4')): os.system('rm -rf {}'.format(join(tar_dir, 'rgb.mp4')))
        cmd = 'ffmpeg -r {} -i {} -vcodec libx264 -crf 28 -pix_fmt yuv420p {}'.format(fps * frame_sample[2], join(tar_dir, 'rgb/%06d.jpg'), join(tar_dir, 'rgb.mp4'))
        os.system(cmd)
        Log.info('Video saved at {}'.format(join(tar_dir, 'rgb.mp4')))

def warp_rgbd(img, depth, conf, tar_path, focal, tar_h=None, tar_w=None, cx=None, cy=None, depth_min=None, depth_max=None, read_depth_func=None, pre_upsample=False, upsample_method='linear'):
    if isinstance(img, str): img = imageio.imread(img)
    if img.shape[0] == tar_h and img.shape[1] != tar_w: 
        img = img[:, :tar_w]
    if img.shape[0] != tar_h and img.shape[1] == tar_w: 
        img = img[:tar_h, :]
    if isinstance(depth, str):
        if read_depth_func is not None: depth = read_depth_func(depth)
        elif depth.endswith('.png'): depth = (np.asarray(imageio.imread(depth)) / 1000.).astype(np.float64)
        elif depth.endswith('.npz'): depth = np.load(depth)['data']
        else: raise ValueError('Unsupported depth format.')
    if isinstance(conf, str): conf = np.asarray(imageio.imread(conf))

    if pre_upsample:
        interpolation = cv2.INTER_LINEAR if upsample_method == 'linear' else cv2.INTER_NEAREST
        depth = cv2.resize(depth, (tar_w, tar_h), interpolation=interpolation)
    if img.shape[0] != tar_h or img.shape[1] != tar_w:
        img = cv2.resize(img, (tar_w, tar_h), interpolation=cv2.INTER_LINEAR)
    img_height, img_width = img.shape[:2]
    fx, fy, cx, cy = focal, focal, img_width/2, img_height/2

    if depth.shape[0] != img_height:
        fx = fx * depth.shape[0] / img_height
        cx = cx * depth.shape[0] / img_height
    
    if depth.shape[1] != img_width:
        fy = fy * depth.shape[1] / img_width
        cy = cy * depth.shape[1] / img_width

    ixt = np.eye(3)
    ixt[0, 0], ixt[1, 1], ixt[0, 2], ixt[1, 2] = fx, fy, cx, cy
    
    color = cv2.resize(img, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)
    points, colors = geo_utils.depth2pcd(depth, ixt, depth_min, depth_max, color, conf=conf)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    def create_transformation_matrix():
        # 向右，向上，向后移动0.5m
        translation = [-0.5, 0.5, 0.5]  
        # 向左，向下转8度
        rotation = o3d.geometry.get_rotation_matrix_from_xyz([-np.deg2rad(-8), -np.deg2rad(-8), 0]) 
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        return transform
    transform = create_transformation_matrix()
    pcd.transform(transform)
    points = np.asarray(pcd.points)
    height, width = depth.shape[:2]
    render_image = np.ones((depth.shape[0], depth.shape[1], 3), dtype=np.uint8) * 255

    cam_points = points @ ixt.T
    depth = cam_points[:, 2]
    image_points = cam_points[:, :2] / depth[:, None]
    x_coords = image_points[:, 0].astype(int)
    y_coords = image_points[:, 1].astype(int)
    valid_indices = (x_coords >= 0) & (x_coords < width) & (y_coords >= 0) & (y_coords < height)
    valid_x_coords = x_coords[valid_indices]
    valid_y_coords = y_coords[valid_indices]
    valid_colors = colors[valid_indices]
    valid_depths = depth[valid_indices]
    sorted_indices = np.argsort(valid_depths)[::-1]
    sorted_x_coords = valid_x_coords[sorted_indices]
    sorted_y_coords = valid_y_coords[sorted_indices]
    sorted_colors = valid_colors[sorted_indices]
    render_image[sorted_y_coords, sorted_x_coords] = sorted_colors

    if render_image.shape[0] != tar_h or render_image.shape[1] != tar_w:
        render_image = cv2.resize(render_image, (tar_w, tar_h), interpolation=cv2.INTER_LINEAR)
    if tar_path is not None:
        os.makedirs(os.path.dirname(tar_path), exist_ok=True)
        imageio.imwrite(tar_path, render_image)
    else:
        return render_image
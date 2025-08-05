import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import PIL
from PIL import Image
import cv2

from trdparties.colmap.read_write_model import read_model
from trdparties.colmap.read_write_model import qvec2rotmat
sys.path.append('.')

scene_to_id = {
    'cc5237fd77': 0,
    '5f99900f09': 1
}

image_size = 384
image_orig_size = 1440
trans_totensor = transforms.Compose([
    transforms.CenterCrop(image_orig_size),
    transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/preview/scannetpp/data')
    parser.add_argument('--output_dir', type=str, default='/mnt/bn/haotongdata/home/linhaotong/projects/monosdf/data/scannetpp')
    parser.add_argument('--scene', type=str, default='cc5237fd77')
    args = parser.parse_args()
    return args

def main(args):
    input_dir = os.path.join(args.input_dir, args.scene)
    output_dir = os.path.join(args.output_dir, 'scan' + str(scene_to_id[args.scene]))
    colmap_dir = join(input_dir, 'iphone', 'colmap')
    cams, images, points3D = read_model(colmap_dir)
    
    cam = cams[1] 
    ixt = np.eye(3)
    ixt[0, 0], ixt[1, 1], ixt[0, 2], ixt[1, 2] = cam.params[:4]
    height, width = cam.height, cam.width
    
    print(ixt) 
    offset_x = (width - image_orig_size) * 0.5
    offset_y = (height - image_orig_size) * 0.5
    ixt[0, 2] -= offset_x
    ixt[1, 2] -= offset_y
    print(ixt)
    resize_factor = image_size / image_orig_size
    ixt[:2, :] *= resize_factor
    print(ixt)
    
    
    poses = [] 
    for idx, image in tqdm(enumerate(images.values())):
        ext = np.eye(4)
        ext[:3, :3] = qvec2rotmat(image.qvec)
        ext[:3, 3] = image.tvec
        c2w = np.linalg.inv(ext)
        poses.append(c2w)
        
    poses = np.stack(poses)
    min_vertices = poses[:, :3, 3].min(axis=0)
    max_vertices = poses[:, :3, 3].max(axis=0)
 
    center = (min_vertices + max_vertices) / 2.
    scale = 2. / (np.max(max_vertices - min_vertices) + 3.)
    print(center, scale)
    scale_mat = np.eye(4).astype(np.float32)
    scale_mat[:3, 3] = -center
    scale_mat[:3 ] *= scale 
    scale_mat = np.linalg.inv(scale_mat)
    
    
    cameras = {}
    for idx, image in tqdm(enumerate(images.values())):
        image_path = join(input_dir, 'iphone', 'rgb', image.name)
        img = Image.open(image_path)
        img_tensor = trans_totensor(img)
        target_image = join(output_dir, 'image', '{:06d}.png'.format(idx))
        os.makedirs(os.path.dirname(target_image), exist_ok=True)
        img_tensor.save(target_image)
        
        mask = (np.ones((image_size, image_size, 3)) * 255.).astype(np.uint8)
        target_image = join(output_dir, 'mask', '{:06d}.png'.format(idx))
        os.makedirs(os.path.dirname(target_image), exist_ok=True)
        cv2.imwrite(target_image, mask)
        
        ext = np.eye(4)
        ext[:3, :3] = qvec2rotmat(image.qvec)
        ext[:3, 3] = image.tvec
        pose = ixt @ ext[:3]
        cameras["scale_mat_%d"%(idx)] = scale_mat
        cameras["world_mat_%d"%(idx)] = pose
        
    np.savez(os.path.join(output_dir, 'cameras.npz'), **cameras)

if __name__ == '__main__':
    args = parse_args()
    main(args)
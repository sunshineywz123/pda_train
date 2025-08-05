import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
sys.path.append('./trdparties/DepthAnything')
# sys.path.append('./trdparties/DepthAnything/torchhub')
# sys.path.append('./trdparties/DepthAnything/torchhub/facebookresearch_dinov2_main')
from PIL import Image
import torch
import glob
from lib.utils.vis_utils import colorize_depth_maps
from trdparties.DepthAnything.depth_anything.dpt import DepthAnything
from trdparties.DepthAnything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import imageio
from torchvision.transforms import Compose
import cv2
import torch.nn.functional as F


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='path to the input folder')
    parser.add_argument('--output', type=str, help='path to output foloder')
    parser.add_argument('--use_disp', action='store_true', help='path to output foloder')
    args = parser.parse_args()
    return args

def predict_depth(img_path, model, use_disp=True):
    raw_image = cv2.imread(img_path)
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    h, w = image.shape[:2]
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
        
    with torch.no_grad():
        depth = model(image)
    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth_numpy = depth.detach().cpu().numpy()
    if not use_disp:
        depth_numpy  = np.clip(depth_numpy, 0.01, None)
        depth_numpy = 1 / depth_numpy
    return depth_numpy

def main(args):
    # repo = "isl-org/ZoeDepth"
    # zoed_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True) 
    # cfg = 'zoedepth_nk' if args.zoe_type == 'nk' else 'zoedepth'
    # conf = get_config(cfg, "infer")
    # zoed_nk = build_model(conf)
    # zoe = zoed_nk.to(DEVICE)
    encoder = 'vitl'
    # depth_anything = DepthAnything.from_pretrained('/mnt/data/home/linhaotong/github_projects/Depth-Anything/metric_depth/checkpoints/depth_anything_{}14.pth'.format(encoder)).to(DEVICE).eval()
    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(DEVICE).eval()
    
    imgs_suffix = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
    img_paths = []
    for suffix in imgs_suffix:
        img_paths.extend(glob.glob(join(args.input, '*'+suffix)))
    img_paths = sorted([img_path for img_path in img_paths if img_path[:1] != '.'])
    dpts = []
    for img_path in tqdm(img_paths):
        depth_numpy = predict_depth(img_path, depth_anything, use_disp=args.use_disp)
        dpts.append(depth_numpy)
    dpts = np.array(dpts)
    dpt_min, dpt_max = dpts.min(), dpts.max()
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(join(args.output, 'npys'), exist_ok=True)
    os.makedirs(join(args.output, 'imgs'), exist_ok=True)
    os.makedirs(join(args.output, 'mp4s'), exist_ok=True)
    
    
    save_imgs = []
    for dpt, img_path in zip(dpts, img_paths):
        npy_path = join(args.output, 'npys/{}'.format(os.path.basename(img_path).split('.')[0]+'.npy'))
        dpt_compressed = np.round(dpt, 5).astype(np.float32)
        np.savez_compressed(npy_path, dpt=dpt_compressed)
        
        dpt_norm = (dpt - dpt_min) / (dpt_max - dpt_min)
        depth_vis = colorize_depth_maps(dpt_norm, 0., 1.)[0].transpose((1, 2, 0))
        save_img = np.concatenate([(depth_vis*255.).astype(np.uint8), np.asarray(Image.open(img_path).convert("RGB"))], axis=1)
        
        img_path = join(args.output, 'imgs/{}'.format(os.path.basename(img_path)))
        imageio.imwrite(img_path, save_img)
        
        save_imgs.append(save_img)
        
    imageio.mimwrite(join(args.output, 'mp4s/output.mp4'), save_imgs, fps=24, quality=7)

if __name__ == '__main__':
    args = parse_args()
    main(args)
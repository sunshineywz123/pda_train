import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
sys.path.append('./trdparties/ZoeDepth')
from PIL import Image
import torch
import glob
from trdparties.ZoeDepth.zoedepth.models.builder import build_model
from trdparties.ZoeDepth.zoedepth.utils.config import get_config
from lib.utils.vis_utils import colorize_depth_maps
import imageio

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='path to the input folder')
    parser.add_argument('--output', type=str, help='path to output foloder')
    parser.add_argument('--zoe_type', type=str, help='path to output foloder', default='nk')
    args = parser.parse_args()
    return args

def predict_depth(img_path, zoe):
    image = Image.open(img_path).convert("RGB")  # load
    depth_numpy = zoe.infer_pil(image)
    return depth_numpy

def main(args):
    # repo = "isl-org/ZoeDepth"
    # zoed_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True) 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = 'zoedepth_nk' if args.zoe_type == 'nk' else 'zoedepth'
    if args.zoe_type == 'nk': cfg = 'zoedepth_nk'
    elif args.zoe_type == 'nyu': cfg = 'zoedepth_nk'
    elif args.zoe_type == 'kitti': cfg = 'zoedepth_nk'
    else: cfg = 'zoedepth'
    conf = get_config(cfg, "infer")
    
    if args.zoe_type == 'kitti':
        conf.pretrained_resource = 'url::https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_K.pt'
    elif args.zoe_type == 'nyu':
        conf.pretrained_resource = 'url::https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_N.pt'
    elif args.zoe_type == 'nk':
        conf.pretrained_resource = 'url::https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_NK.pt'
    else:
        import ipdb; ipdb.set_trace()
    # zoed_nk = build_model(conf)
    repo = "isl-org/ZoeDepth"
    zoed_nk = torch.hub.load(repo, "ZoeD_K", pretrained=True)
    zoe = zoed_nk.to(DEVICE)
    
    
    imgs_suffix = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
    img_paths = []
    for suffix in imgs_suffix:
        img_paths.extend(glob.glob(join(args.input, '*'+suffix)))
    img_paths = sorted([img_path for img_path in img_paths if img_path[:1] != '.'])
    dpts = []
    for img_path in tqdm(img_paths):
        depth_numpy = predict_depth(img_path, zoe)
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
import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
sys.path.append('./trdparties/Marigold')
from  trdparties.Marigold.marigold import MarigoldPipeline
from PIL import Image
import torch
import glob
from lib.utils.vis_utils import colorize_depth_maps
import imageio
import cv2
import torch.nn.functional as F


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='path to the input folder')
    parser.add_argument('--output', type=str, help='path to output foloder')
    # parser.add_argument('--use_disp', type=bool, help='path to output foloder', default='nk')
    args = parser.parse_args()
    return args

def predict_depth(img_path, model, use_disp=True):
    input_image = Image.open(img_path)
    
    pipe_out = model(
                input_image,
                denoising_steps=10,
                ensemble_size=5,
                processing_res=768,
                match_input_res=True,
                batch_size=1,
                color_map='Spectral',
                show_progress_bar=False,
    )
    depth_pred: np.ndarray = pipe_out.depth_np
    return depth_pred

def main(args):
    # repo = "isl-org/ZoeDepth"
    # zoed_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True) 
    # cfg = 'zoedepth_nk' if args.zoe_type == 'nk' else 'zoedepth'
    # conf = get_config(cfg, "infer")
    # zoed_nk = build_model(conf)
    # zoe = zoed_nk.to(DEVICE)
    dtype = torch.float32
    checkpoint_path = '/mnt/data/home/linhaotong/research_projects/pl_htcode/data/pl_htcode/Marigold/checkpoint/Marigold_v1_merged_2'
    pipe = MarigoldPipeline.from_pretrained(checkpoint_path, torch_dtype=dtype)
    pipe = pipe.to(DEVICE)
    
    imgs_suffix = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
    img_paths = []
    for suffix in imgs_suffix:
        img_paths.extend(glob.glob(join(args.input, '*'+suffix)))
    img_paths = sorted([img_path for img_path in img_paths if img_path[:1] != '.'])
    dpts = []
    for img_path in tqdm(img_paths):
        depth_numpy = predict_depth(img_path, pipe)
        dpts.append(depth_numpy)
    dpts = np.array(dpts)
    dpt_min, dpt_max = dpts.min(), dpts.max()
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(join(args.output, 'npys'), exist_ok=True)
    os.makedirs(join(args.output, 'imgs'), exist_ok=True)
    os.makedirs(join(args.output, 'mp4s'), exist_ok=True)
    
    
    save_imgs = []
    for dpt, img_path in zip(dpts, img_paths):
        if len(os.path.basename(img_path).split('.')) == 2:
            npy_path = join(args.output, 'npys/{}'.format(os.path.basename(img_path).split('.')[0]+'.npy'))
        else:
            npy_path = join(args.output, 'npys/{}'.format('.'.join(os.path.basename(img_path).split('.')[:-1])+'.npy'))
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
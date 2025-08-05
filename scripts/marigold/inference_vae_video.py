import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')

from diffusers import AutoencoderKL
import torch
from PIL import Image
from scripts.marigold.inference_vae_latent import load_tensor_from_image, encode_rgb, vis_tensor
import torch.nn.functional as F
import imageio
import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default='/mnt/data/home/linhaotong/research_projects/pl_htcode/data/pl_htcode/dynibar_release/kid-running/dense/images_512x288')
    parser.add_argument('--input_vae_path', type=str, default='data/pl_htcode/Marigold/checkpoint/Marigold_v1_merged_2/vae')
    parser.add_argument('--output_path', type=str, default='test.mp4')
    args = parser.parse_args()
    return args

def main(args):
    vae = AutoencoderKL.from_pretrained(args.input_vae_path)
    vae.eval()
    vae.requires_grad_(False)
    vae.to('cuda')
    
    rgb_paths = sorted(os.listdir(args.input_folder))
    outputs = []
    for rgb_path in rgb_paths:
        img_path = join(args.input_folder, rgb_path)
        img_tensor = load_tensor_from_image(img_path).cuda() * 2 - 1.
        latent = encode_rgb(img_tensor, vae)
        latent = latent.permute(0, 2, 3, 1)
        latent = latent[0]
        latent_vis = vis_tensor(latent)
        img_vis = img_tensor * 0.5 + 0.5
        img_vis = F.interpolate(img_vis, size=latent_vis.shape[:2], mode='bilinear')[0].permute(1, 2, 0).detach().cpu().numpy()
        latent_vis = latent_vis.detach().cpu().numpy()
        
        img_vis = cv2.resize(img_vis, (512, 288), interpolation=cv2.INTER_AREA)
        latent_vis = cv2.resize(latent_vis, (512, 288), interpolation=cv2.INTER_AREA)
        outputs.append(np.concatenate([img_vis, latent_vis], axis=0))
    imageio.mimwrite(args.output_path, outputs, fps=24)

if __name__ == '__main__':
    args = parse_args()
    main(args)
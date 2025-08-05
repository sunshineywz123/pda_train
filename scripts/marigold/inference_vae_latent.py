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
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_img_path', type=str, default='/mnt/data/home/linhaotong/research_projects/pl_htcode/data/pl_htcode/dynibar_release/kid-running/dense/images_512x288/00000.png')
    parser.add_argument('--input_vae_path', type=str, 
                        default='data/pl_htcode/Marigold/checkpoint/Marigold_v1_merged_2/vae')
    parser.add_argument('--output_img_path', type=str, default='test.jpg')
    args = parser.parse_args()
    return args

def vis_tensor(tensor: torch.Tensor, k: int = 3):
    assert(len(tensor.shape) == 3)
    scaled_data = tensor.reshape(-1, tensor.shape[-1])
    U, S, V = torch.svd(scaled_data)
    principal_components = V[:, :k]
    projected_data = torch.mm(scaled_data, principal_components)
    min_val, max_val = projected_data.min(), projected_data.max()
    projected_data = (projected_data - min_val) / (max_val - min_val)
    return projected_data.reshape(tensor.shape[0], tensor.shape[1], k)

def main(args):
    # define vae_model
    vae = AutoencoderKL.from_pretrained(args.input_vae_path)
    vae.eval()
    vae.requires_grad_(False)
    vae.to('cuda')
    
    img_tensor = load_tensor_from_image(args.input_img_path).cuda() * 2 - 1.
    
    latent = encode_rgb(img_tensor, vae)
    latent = latent.permute(0, 2, 3, 1)
    latent = latent[0]
    scaled_data = latent.reshape(-1, latent.shape[-1])
    U, S, V = torch.svd(scaled_data)
    k = 3
    principal_components = V[:, :k]
    projected_data = torch.mm(scaled_data, principal_components)
    latent_vis = projected_data.reshape(latent.shape[0], latent.shape[1], k)
    # plt.imshow(latent_vis.detach().cpu().numpy())
    # plt.savefig(args.output_img_path)
    # plt.imshow(latent[:, :, :3])
    # plt.savefig(args.output_img_path)
    # import ipdb; ipdb.set_trace()
    

def load_tensor_from_image(img_path: str, resize: int = 768):
    """
    Args:
        img_path (str): path to image
    Returns:
        img (torch.Tensor): shape (H, W, 3)
    """
    img = Image.open(img_path)
    img_size = img.size
    if max(img_size) != resize:
        img = img.resize((int(img_size[0] * resize / max(img_size)), int(img_size[1] * resize / max(img_size))))
    img = torch.from_numpy(np.array(img))
    img = img.permute(2, 0, 1).float() / 255.
    return img[None]

def encode_rgb(rgb_in: torch.Tensor, vae: AutoencoderKL, rgb_latent_scale_factor: float = 0.18215):
    """
    Args:
        rgb_in (torch.Tensor): shape (H, W, 3)
    Returns:
        latent (torch.Tensor): shape (1, 256)
    """
    h = vae.encoder(rgb_in)
    return h
    moments = vae.quant_conv(h)
    mean, logvar = torch.chunk(moments, 2, dim=1)
    rgb_latent = mean * rgb_latent_scale_factor
    return rgb_latent

if __name__ == '__main__':
    args = parse_args()
    main(args)
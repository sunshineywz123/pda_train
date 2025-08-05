import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')

from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb_input', type=str, default='data/pl_htcode/test/dog.png')
    parser.add_argument('--msk_input', type=str, default='data/pl_htcode/test/mask.png')
    parser.add_argument('--rgb_output', type=str, default='data/pl_htcode/test/output.png')
    args = parser.parse_args()
    return args

def main(args):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")
    prompt = ""
    #image and mask_image should be PIL images.
    #The mask structure is white for inpainting and black for keeping as is
    image = Image.open(args.rgb_input)
    mask_image = Image.open(args.msk_input)
    
    image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
    if not os.path.exists(os.path.dirname(args.rgb_output)):
        os.makedirs(os.path.dirname(args.rgb_output))
    image.save(args.rgb_output)

if __name__ == '__main__':
    args = parse_args()
    main(args)
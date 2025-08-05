import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
import cv2
import imageio

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/data/home/linhaotong/projects/pl_htcode/data/pl_htcode/vdw/000025')
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    return args


def warp_image(img_left, disp_left):
    """
    Warps the left image to the perspective of the right image using the disparity map.
    
    Parameters:
    - img_left: The left image as a numpy array (height, width, channels).
    - disp_left: The disparity map for the left image as a numpy array (height, width).
                 The disparity values should indicate how much each pixel in the left image
                 needs to move to the right to match the corresponding pixel in the right image.
    
    Returns:
    - img_right: The warped right image as a numpy array (height, width, channels).
    """
    
    h, w = disp_left.shape[:2]
    # Create the output image, which will be filled with the warped left image
    img_right = np.zeros_like(img_left)
    
    # Generate meshgrid for pixel coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    # Calculate the new coordinates based on the disparity map
    new_x = (x - disp_left).clip(0, w-1).astype(np.float32)
    new_y = y.astype(np.float32)
    
    # Warp the left image to the right perspective
    img_right = cv2.remap(img_left, new_x, new_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img_right

def main(args):
    shift_l, scale_l, shift_r, scale_r = np.loadtxt(join(args.input, 'shift_scale_lr.txt'))
    
    img_left = imageio.imread(join(args.input, 'left/frame_000000.png'))
    gt_left = imageio.imread(join(args.input, 'left_gt/frame_000000.png'))
    # msk_left = imageio.imread(join(args.input, 'left_mask/frame_000000.png'))
    img_right = imageio.imread(join(args.input, 'right/frame_000000.png'))
    
    disp_left = (gt_left - shift_l) / scale_l
    
    # warp left image to right image according to disparity
    warp_img_right = warp_image(img_left, disp_left)

    import ipdb; ipdb.set_trace()
    
        
    
    # warp right image to left image
    
    
    
    pass

if __name__ == '__main__':
    args = parse_args()
    main(args)
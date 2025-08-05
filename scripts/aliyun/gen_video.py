import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
import imageio
import copy
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    return args

def paste_a2b(a, b, bkgd):
    msk = (np.abs(a-bkgd).sum(axis=-1) > 0.03).astype(np.float32) # 255 * 0.01
    # imageio.imwrite('debug.jpg', (msk*255.).astype(np.uint8))
    ret = a * msk[..., None] + b * (1 - msk[..., None])
    return ret

def paste_a2b_withmsk(a, b, msk):
    return b * (1-msk[..., None]) + a * msk[..., None]

def main(args):
    input_dir_ = 'data/pl_htcode/links/easymocap/seq3_round00'
    cam_id = int(os.environ['cam_id'])
    key_frames = [20, 40, 60, 80]
    output_path = 'output_{:04}.mp4'.format(cam_id)
    
    bkgd_image = (imageio.imread(join(input_dir_, 'bkgd/{:04d}.jpg'.format(cam_id))) / 255.)
    input_dir = join(input_dir_, 'images/{:04d}'.format(cam_id))
    matting_dir = join(input_dir_, 'bgmtv2/{:04d}'.format(cam_id))
    mask_dir = join(input_dir_, 'mask/{:04d}'.format(cam_id))
    
    imgs = []
    reserve_img = None
    for i in tqdm(range(key_frames[-1] + key_frames[0])):
        img = (imageio.imread(join(input_dir, f'{i:06d}.jpg')) / 255.)
        init_matting = (imageio.imread(join(matting_dir, f'{i:06d}.jpg')) / 255.)
        # mask = (imageio.imread(join(mask_dir, f'{i:06d}.png')) / 255.)
        kernel = np.ones((3,3),np.float32)
        mask = cv2.erode((init_matting>0).astype(np.uint8), kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
        final_matting = np.zeros_like(init_matting)
        final_matting[y:y+h, x:x+w] = init_matting[y:y+h, x:x+w]
        matting = final_matting
        if i in key_frames:
            if reserve_img is None:
                reserve_img = img
            else:
                # reserve_img = paste_a2b(img, reserve_img, bkgd_image)
                reserve_img = paste_a2b_withmsk(img, reserve_img, matting)
        if reserve_img is not None:
            # img = paste_a2b(img, reserve_img, bkgd_image)
            img = paste_a2b_withmsk(img, reserve_img, matting)
        img = (img * 255.).astype(np.uint8)
        imgs.append(img)
        imageio.imwrite('{:04d}.jpg'.format(i), img)
    imageio.mimwrite(output_path, imgs, fps=15)

if __name__ == '__main__':
    args = parse_args()
    main(args)
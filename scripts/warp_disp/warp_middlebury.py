import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
import cv2
import imageio
import re
from scripts.warp_disp.utils import warp_image

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

def process_one_scene(input_dir, scene, output_dir, args):
    input_path = join(input_dir, scene)
    img_left = (imageio.imread(join(input_dir, scene, 'im0.png')) / 255.).astype(np.float32)
    disp_left, disp_left_scale = read_pfm(join(input_dir, scene, 'disp0GT.pfm'))
    warp_img_right, msk_right = warp_image(img_left, disp_left, weight_thresh=args.weight_thresh, disp_thresh=args.disp_thresh)
    
    os.makedirs(join(output_dir, scene), exist_ok=True)
    os.system('cp {} {}'.format(join(input_path, 'im0.png'), join(output_dir, scene, 'im0.png')))
    os.system('cp {} {}'.format(join(input_path, 'im1.png'), join(output_dir, scene, 'im1.png')))
    imageio.imwrite(join(output_dir, scene, 'warp_im1_gtdisp.png'), (np.clip(warp_img_right, 0., 1.) * 255).astype(np.uint8))
    imageio.imwrite(join(output_dir, scene, 'warp_im1_gtdisp_mask.png'), msk_right.astype(np.uint8) * 255)
    
    warp_img_right, msk_right = warp_image(img_left, disp=None, disp_align=disp_left, weight_thresh=args.weight_thresh, disp_thresh=args.disp_thresh)
    imageio.imwrite(join(output_dir, scene, 'warp_im1_preddisp.png'), (np.clip(warp_img_right, 0., 1.) * 255).astype(np.uint8))
    imageio.imwrite(join(output_dir, scene, 'warp_im1_predisp_mask.png'), msk_right.astype(np.uint8) * 255)



def main(args):
    for scene in tqdm(os.listdir(args.input_dir)):
        process_one_scene(args.input_dir, scene, args.output_dir, args)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/mnt/data/Datasets/middlebury/MiddEval3/trainingQ')
    parser.add_argument('--output_dir', type=str, default='/nas/home/linhaotong/middlebury')
    parser.add_argument('--weight_thresh', type=float, default=0.01)
    parser.add_argument('--disp_thresh', type=float, default=2)
    args = parser.parse_args()
    return args
        
if __name__ == '__main__':
    args = parse_args()
    main(args)
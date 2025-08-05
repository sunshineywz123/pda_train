import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
import imageio
sys.path.append('.')
import cv2

# parttern = {300: [(65, 'static'), 
#                  (20, 'up_half'), 
#                  (50, 'static'), 
#                  (30, 'down_whole'), 
#                  (50, 'static'),
#                  (20, 'up_half'), 
#                  (65, 'static')]}
parttern = {300: [(50, 'static'), 
                 (20, 'up_half'), 
                 (30, 'static'), 
                 (40, 'down_whole'), 
                 (20, 'up_half'), 
                 (50, 'static'),
                 (20, 'up_half'), 
                 (30, 'static'), 
                 (20, 'down_half'), 
                 (20, 'static')]}

def get_line_func(h, w, height):
    b = height * h + h * 0.5
    a = - h / w
    return a, b

def get_mask(h, w, height):
    mask = np.zeros((h, w), dtype=np.uint8)
    a, b = get_line_func(h, w, height)
    for y in range(h):
        for x in range(w):
            if y < a * x + b:
                mask[y, x] = 1
    return mask


def cosine_interpolation(init_height, tar_height, length):
    # Generate an array of indices from 0 to length-1
    indices = np.arange(length)
    # Calculate the cosine interpolation
    interpolated_values = init_height + (tar_height - init_height) * (1 - np.cos(np.pi * indices / (length - 1))) / 2
    return interpolated_values


def exponential_interpolation(init_height, tar_height, length, rate=2):
    # Generate an array of indices from 0 to length-1
    indices = np.linspace(0, 1, length)
    # Calculate the exponential interpolation
    interpolated_values = init_height + (tar_height - init_height) * (1 - np.exp(-rate * indices)) / (1 - np.exp(-rate))
    
    return interpolated_values


def merge_img(img1, img2, height=0.5, line_width=5, name1='ours', name2='lidar'):
    '''
    line_width: int, pixels

    '''
    # new_img = np.ones_like(img1) * 255
    h, w = img1.shape[:2]
    a, b = get_line_func(h, w, height)

    mask1 = np.zeros((h, w), dtype=np.uint8)
    mask2 = np.zeros((h, w), dtype=np.uint8)
    line_mask = np.zeros((h, w), dtype=np.uint8)

    y, x = np.ogrid[:h, :w]
    distance = np.abs(a * x - y + b) / np.sqrt(a**2 + 1)
    line_mask = distance < line_width / 2.0
    mask1 = y < a * x + b
    mask2 = ~mask1
    new_img = np.zeros_like(img1)
    new_img[mask1 == 1] = img1[mask1 == 1]
    new_img[mask2 == 1] = img2[mask2 == 1]
    new_img[line_mask == 1] = 255

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_size1, _ = cv2.getTextSize(name1, font, font_scale, font_thickness)
    text_size2, _ = cv2.getTextSize(name2, font, font_scale, font_thickness)

    # Coordinates for the text backgrounds
    text_origin1 = (10, text_size1[1] + 10)
    text_origin2 = (w - text_size2[0] - 10, h - 10)

    # Draw gray rectangles
    cv2.rectangle(new_img, (text_origin1[0] - 5, text_origin1[1] - text_size1[1] - 5), 
                  (text_origin1[0] + text_size1[0] + 5, text_origin1[1] + 5), (128, 128, 128), cv2.FILLED)
    cv2.rectangle(new_img, (text_origin2[0] - 5, text_origin2[1] - text_size2[1] - 5), 
                  (text_origin2[0] + text_size2[0] + 5, text_origin2[1] + 5), (128, 128, 128), cv2.FILLED)

    # Put white text on top of gray rectangles
    cv2.putText(new_img, name1, text_origin1, font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    cv2.putText(new_img, name2, text_origin2, font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    return new_img

def generate_heights(parttern):
    height = 0.5
    heights = []
    for item in parttern:
        length, type = item
        if type == 'static':
            heights.extend([height] * length)
        elif type == 'up_half':
            init_height = height
            tar_height = height - 0.5
            height = tar_height
            heights_segment = exponential_interpolation(init_height, tar_height, length)
            heights.extend(heights_segment)
        elif type == 'down_whole':
            init_height = height
            tar_height = height + 1.
            height = tar_height
            heights_segment = exponential_interpolation(init_height, tar_height, length)
            heights.extend(heights_segment)
        elif type == 'down_half':
            init_height = height
            tar_height = height + 0.5
            height = tar_height
            heights_segment = exponential_interpolation(init_height, tar_height, length)
            heights.extend(heights_segment)
        else:
            raise ValueError
    return heights

def main(args):
    # read imgs_1
    # read imgs_2
    heights = generate_heights(parttern[300])
    imgs_1 = [np.asarray(imageio.imread(join(args.input1, img_name))) for img_name in tqdm(sorted(os.listdir(args.input1)))]
    imgs_2 = [np.asarray(imageio.imread(join(args.input2, img_name))) for img_name in tqdm(sorted(os.listdir(args.input2)))]

    output_imgs = []
    for img1, img2, height in tqdm(zip(imgs_1, imgs_2, heights)):
        output_imgs.append(merge_img(img1, img2, height, line_width=5, name1=args.name1, name2=args.name2))
    imageio.mimwrite(args.output, output_imgs, fps=24)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input1', type=str, default='/mnt/bn/haotongdata/home/linhaotong/projects/2dgs/output/20240702_statue3_depth_l10_july_hypersim_baseline0717_minmax_human_split15/traj/ours_30000/renders')
    parser.add_argument('--input2', type=str, default='/mnt/bn/haotongdata/home/linhaotong/projects/2dgs/output/20240702_statue3_lidar_l10_split15/traj/ours_30000/renders')
    parser.add_argument('--name1', type=str, default='ours')
    parser.add_argument('--name2', type=str, default='lidar')
    parser.add_argument('--output', type=str, default='data/pl_htcode/demos/ours_lidar_15views.mp4')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)
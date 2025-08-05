import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
import imageio
sys.path.append('.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video_path', type=str, help='path to the input video or the folder of video frames', default='data/pl_htcode/dynibar_release/kid-running/dense/images_512x288')
    parser.add_argument('--num_step', type=int, help='number of frames to skip', default=5)
    parser.add_argument('--total_frames', type=int, help='total number of frames to extract', default=8)
    parser.add_argument('--shift_pixels', type=int, help='number of pixels to shift the frame', default=12)
    parser.add_argument('--add_white_border', type=int, help='number of pixels to add white border', default=3)
    parser.add_argument('--output_path', type=str, help='path to the output folder', default='output.png')
    args = parser.parse_args()
    return args

def read_frames(args):
    if os.path.isdir(args.input_video_path):
        frames = sorted([join(args.input_video_path, f) for f in os.listdir(args.input_video_path) if (f.endswith('.png') or f.endswith('.jpg')) and f[:1] != '.'])
    else:
        import ipdb; ipdb.set_trace()
    frames = frames[::args.num_step][:args.total_frames]
    images = np.asarray([imageio.imread(frame) for frame in frames])#[:, :, :512]
    if args.add_white_border != 0:
        images = np.pad(images, ((0, 0), (args.add_white_border, args.add_white_border), (args.add_white_border, args.add_white_border), (0, 0)), mode='constant', constant_values=255)
    return images
    
def generate_image(frames, args):
    shift_pixels = args.shift_pixels
    total_frames = min(args.total_frames, len(frames))
    board = np.ones((frames.shape[1] + shift_pixels * (total_frames - 1), frames.shape[2] + shift_pixels * (total_frames - 1), 3), dtype=np.uint8) * 255
    
    for i in range(total_frames):
        board[i*shift_pixels:i*shift_pixels+frames.shape[1], i*shift_pixels:i*shift_pixels+frames.shape[2], :] = frames[i]
    return board

def main(args):
    frames = read_frames(args)
    img = generate_image(frames, args)
    imageio.imwrite(args.output_path, img)

if __name__ == '__main__':
    args = parse_args()
    main(args)
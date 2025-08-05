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
    parser.add_argument('--output_video_dir', type=str, help='number of frames to skip', default='output')
    args = parser.parse_args()
    return args

def read_frames(args):
    if os.path.isdir(args.input_video_path):
        frames = sorted([join(args.input_video_path, f) for f in os.listdir(args.input_video_path) if (f.endswith('.png') or f.endswith('.jpg')) and f[:1] != '.'])
    else:
        import ipdb; ipdb.set_trace()
    images = np.asarray([imageio.imread(frame) for frame in frames])#[:, :, :512]
    frames = [os.path.basename(frame) for frame in frames]
    return frames, images
    
def main(args):
    frames, images = read_frames(args)
    os.makedirs(join(args.output_video_dir, 'noise'), exist_ok=True)
    os.makedirs(join(args.output_video_dir, 'noisy_images'), exist_ok=True)
    
    for frame, image in tqdm(zip(frames, images)):
        noise = (np.random.normal(0, 1, image.shape) * 255).astype(np.uint8)
        noisy_image = (noise * 0.5 + image * 0.5).astype(np.uint8)
        imageio.imsave(join(args.output_video_dir, 'noise', frame), noise)
        imageio.imsave(join(args.output_video_dir, 'noisy_images', frame), noisy_image)

if __name__ == '__main__':
    args = parse_args()
    main(args)
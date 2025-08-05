import os
from os.path import join
import argparse
import shutil
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')

def main(args):
    root = '/mnt/data/home/linhaotong/datasets/mycapture4d/20240728'
    output = '/mnt/data/home/linhaotong/datasets/mycapture4d/20240728_seq2'
    videos = ['30', '31', '32', '33', '34']
    double_speed = [True, False, False, False, False]
    seq = 'seq2'

    for video in tqdm(videos):
        os.makedirs(join(root, video, seq, 'rgb'), exist_ok=True)
        cmd = 'ffmpeg -i {} -start_number 0 -q:v 2 {}/%06d.jpg'.format(join(root, video, seq, 'rgb.mp4'), 
                                                                          join(root, video, seq, 'rgb'))
        # os.system(cmd)

    # syncs = [4407, 2121, 2801, 2058, 2108]
    syncs = [6342, 3909, 4381, 4096, 4503]
    default_video = 1
    background = 3608
    frame_range = [2108, 3508]

    backgrounds = []
    output_frames = []
    for idx, sync in enumerate(syncs):
        backgrounds.append(
            sync - (syncs[default_video] - background) * (2 if double_speed[idx] else 1)
        )
        frames = []
        for frame in range(frame_range[0], frame_range[1]):
            frames.append(
                sync - (syncs[default_video] - frame) * (2 if double_speed[idx] else 1)
            )
        output_frames.append(frames)

    os.makedirs(join(output, 'rgb'), exist_ok=True)
    for idx, frames in enumerate(output_frames):
        # src = join(root, videos[idx], seq, 'rgb', f'{backgrounds[idx]:06d}.jpg')
        # dst = join(output, 'bkgd', f'{videos[idx]}.jpg')
        # os.makedirs(os.path.dirname(dst), exist_ok=True)
        # shutil.copy(src, dst)

        # src = join(root, videos[idx], seq, 'depth', f'{backgrounds[idx]:06d}.png')
        # dst = join(output, 'bkgd_depth', f'{videos[idx]}.png')
        # os.makedirs(os.path.dirname(dst), exist_ok=True)
        # shutil.copy(src, dst)

        # src = join(root, videos[idx], seq, 'confidence', f'{backgrounds[idx]:06d}.png')
        # dst = join(output, 'bkgd_conf', f'{videos[idx]}.png')
        # os.makedirs(os.path.dirname(dst), exist_ok=True)
        # shutil.copy(src, dst)
        for frame_id, frame in tqdm(enumerate(frames)):
            src = join(root, videos[idx], seq, 'rgb', f'{frame:06d}.jpg')
            dst = join(output, 'images', videos[idx], f'{frame_id:06d}.jpg')
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)

            src = join(root, videos[idx], seq, 'depth', f'{frame:06d}.png')
            dst = join(output, 'depth', videos[idx], f'{frame_id:06d}.png')
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/data/home/linhaotong/datasets/mycapture4d/20240728')
    parser.add_argument('--output', type=str, default='/mnt/data/home/linhaotong/datasets/mycapture4d/20240728_seq1')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)
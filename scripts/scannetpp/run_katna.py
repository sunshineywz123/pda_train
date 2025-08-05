import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
# import katna

from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter
import ntpath

# # # For windows, the below if condition is must.
# # if __name__ == "__main__":

# #   #instantiate the video class
# #   vd = Video()

# #   #number of key-frame images to be extracted
# #   no_of_frames_to_return = 3

# #   #Input Video directory path
# #   #All .mp4 and .mov files inside this directory will be used for keyframe extraction)
# #   videos_dir_path = os.path.join(".", "tests","data")

# #   diskwriter = KeyFrameDiskWriter(location="selectedframes")

# #   vd.extract_keyframes_from_videos_dir(
# #        no_of_frames=no_of_frames_to_return, dir_path=videos_dir_path,
# #        writer=diskwriter
# #   )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mp4_path', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/98b4ec142f/iphone/rgb.mp4')
    parser.add_argument('--output', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/98b4ec142f/iphone/keyframes')
    args = parser.parse_args()
    return args

def main(args):
    vd = Video()
    no_of_frames_to_returned = 12
    video_file_path = args.input_mp4_path
    diskwriter = KeyFrameDiskWriter(location=args.output)
    vd.extract_video_keyframes(
       no_of_frames=no_of_frames_to_returned, file_path=video_file_path,
       writer=diskwriter
    )

if __name__ == '__main__':
    args = parse_args()
    main(args)
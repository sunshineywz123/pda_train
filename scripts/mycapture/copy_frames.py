import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
from lib.utils.parallel_utils import parallel_execution

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    return args

def copy_frames(input_dir):
    files = os.listdir(input_dir)
    files = [f for f in files if f.endswith('.jpg')]
    files = sorted(files)
    files.reverse()
    for f in tqdm(files):
        src_path = join(input_dir, f)
        num = int(f.split('.')[0])
        if num != 0: 
            tar_path = join(input_dir, '{:06d}.jpg'.format(2 * num))
            os.system('cp -r ' + src_path + ' ' + tar_path)
            tar_path = join(input_dir, '{:06d}.jpg'.format(2 * num + 1))
            os.system('mv ' + src_path + ' ' + tar_path)
        else:
            tar_path = join(input_dir, '{:06d}.jpg'.format(2 * num + 1))
            os.system('cp -r ' + src_path + ' ' + tar_path)

def process_seq_video(input_tuple):
    input_dir, output_dir, seq, video = input_tuple
    input_video = join(input_dir, seq, video + '.mp4')
    output_temp_dir = join(output_dir, seq, video)
    os.makedirs(output_temp_dir, exist_ok=True)
    os.system(f'ffmpeg -i {input_video} -start_number 0 -q:v 1 {output_temp_dir}/%06d.jpg')
    copy_frames(output_temp_dir)
    os.system('ffmpeg -framerate 60 -i {}/%06d.jpg -vframes 6000 -c:v libx264 -pix_fmt yuv420p {}/{}/{}.mp4'.format(output_temp_dir, output_dir, seq, video))
    os.system('rm -rf ' + output_temp_dir)

def main(args):
    output_dir = 'output'
    input_dir = '/home/linhaotong/0704'
    seqs = ['seq1', 'seq2', 'seq3']
    videos = ['30', '31', '32', '33']
    input_tuples = []
    for seq in seqs:
        for video in videos:
            input_tuples.append((input_dir, output_dir, seq, video))
    parallel_execution(
        input_tuples,
        process_seq_video,
        num_processes=12,
        print_progress=True
    )

if __name__ == '__main__':
    args = parse_args()
    main(args)
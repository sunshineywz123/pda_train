import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
import imageio
import cv2

def main(args):
    cameras = ['30', '31', '32', '33']
    rotate = ['30', '31']
    syncs = [4742, 4899, 4082, 5113]
    fpss = [30, 30, 30, 60]

    # for idx, camera in enumerate(cameras):
    #     if camera in rotate:
    #         cmd = 'ffmpeg -i {} -vf "transpose=1,transpose=1" -crf 18 {}'.format(join(args.input, f'{camera}', 'rgb.mp4'),
    #                                                                     join(args.input, f'{camera}', 'rgb_transpose.mp4'))
    #         os.system(cmd)
    
    # for idx, camera in enumerate(cameras):
    #     # if camera in rotate:
    #     #     source_mp4 = join(args.input, f'{camera}', 'rgb_transpose.mp4')
    #     # else:
    #     source_mp4 = join(args.input, f'{camera}', 'rgb.mp4')
    #     output_dir = join(args.input, 'seq2_mocap', 'images', camera)
    #     os.makedirs(output_dir, exist_ok=True)
    #     if fpss[idx] == 60:
    #         cmd = f'ffmpeg -i {source_mp4} -vf "select=\'between(n\,{syncs[idx]}\,{syncs[idx]+4499})\',setpts=N/FRAME_RATE/TB" -vsync vfr -start_number 0 -frame_pts true -q:v 2 {output_dir}/%06d.jpg'
    #     elif fpss[idx] == 30:
    #         cmd = f'ffmpeg -i {source_mp4} -vf "select=\'between(n\,{syncs[idx]}\,{syncs[idx]+2249})\',setpts=N/FRAME_RATE/TB" -vsync vfr -start_number 0 -frame_pts true -q:v 2 {output_dir}/%06d.jpg'
    #     os.system(cmd)

    # for idx, camera in enumerate(cameras):
    #     if fpss[idx] == 60:
    #         continue
    #     output_dir = join(args.input, 'seq2_mocap', 'images', camera)
    #     img_files = sorted(os.listdir(output_dir))[::-1]
    #     for img_file in img_files:
    #         img_idx = int(img_file.split('.')[0])
    #         tar_img_idx = 2 * img_idx
    #         if tar_img_idx != 0:
    #             os.system('mv {} {}'.format(join(output_dir, img_file), join(output_dir, f'{tar_img_idx:06d}.jpg')))
    #         tar_img_idx2 = 2 * img_idx + 1
    #         os.system('ln -s {} {}'.format(join(output_dir, f'{tar_img_idx:06d}.jpg'), join(output_dir, f'{tar_img_idx2:06d}.jpg')))


    for idx, camera in enumerate(cameras):
        output_dir = join(args.output, 'seq2_mocap', 'depth', camera)
        os.makedirs(output_dir, exist_ok=True)
        if camera in rotate:
            continue
            for i in tqdm(range(2250)):
                source_path = join(args.input, f'{camera}', 'depth', f'{syncs[idx]+i:06d}.png')
                tar_path = join(args.output, 'seq2_mocap', 'depth', camera, f'{2*i:06d}.png')
                img = imageio.imread(source_path)
                img = cv2.rotate(img, cv2.ROTATE_180)
                imageio.imwrite(tar_path, img)
                tar_path2 = join(args.output, 'seq2_mocap', 'depth', camera, f'{2*i+1:06d}.png')
                os.system('ln -s {} {}'.format(os.path.basename(tar_path), tar_path2))
        else:
            if fpss[idx] == 60:
                for i in tqdm(range(4500)):
                    sourec_path = join(args.input, f'{camera}', 'depth', f'{syncs[idx]+i:06d}.png')
                    tar_path = join(args.output, 'seq2_mocap', 'depth', camera, f'{i:06d}.png')
                    os.system('cp {} {}'.format(sourec_path, tar_path))
            else:
                for i in tqdm(range(2250)):
                    source_path = join(args.input, f'{camera}', 'depth', f'{syncs[idx]+i:06d}.png')
                    tar_path = join(args.output, 'seq2_mocap', 'depth', camera, f'{2*i:06d}.png')
                    os.system('cp {} {}'.format(source_path, tar_path))
                    tar_path2 = join(args.output, 'seq2_mocap', 'depth', camera, f'{2*i+1:06d}.png')
                    os.system('ln -s {} {}'.format(os.path.basename(tar_path), tar_path2))
        
def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input', type=str, default='/mnt/bn/haotongdata/Datasets/mycapture_stray_scanner/seq2_raw')
    parser.add_argument('--input', type=str, default='/Users/linhaotong/ResearchData/Projects/2025_CVPR/mycapture/stray_scanner/seq2_raw')
    parser.add_argument('--output', type=str, default='.')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)

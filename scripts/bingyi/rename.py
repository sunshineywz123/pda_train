import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
import cv2

def main(args):
    names = sorted(os.listdir(args.input))
    for i, name in enumerate(tqdm(names)):
        img = cv2.imread(join(args.input, name))
        cv2.imwrite(join(args.output, f'frame_{i:06d}.jpg'), img)
        print(f'{name} -> frame_{i:06d}.jpg')
        # os.rename(join(args.input, name), join(args.output, f'frame_{i:06d}.jpg'))
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/bn/haotongdata/test/synthetic2/new_model/RGB')
    parser.add_argument('--output', type=str, default='/mnt/bn/haotongdata/test/synthetic2/new_model/images')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)
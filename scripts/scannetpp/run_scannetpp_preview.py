
import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
import glob

np.random.seed(42)

def main(args):
    scenes = os.listdir(args.input)
    os.makedirs(args.output, exist_ok=True)
    for scene in tqdm(scenes):
        if not os.path.exists(join(args.input, scene, 'test_preds')):
            print(scene)
            # /mnt/bn/haotongdata/home/linhaotong/projects/zipnerf-pytorch/exp/scannetpp_all_0610/104acbf7d2/checkpoints/050000/model.safetensors
            # if os.path.exists(join(args.input, scene, 'checkpoints/050000/model.safetensors')):
            #     print('Prediction', scene)
            # elif os.path.exists(join(args.input, scene, 'checkpoints')):
            #     print('One more check', scene)
            # else:
            #     print('debug', scene)
            # continue
        rgbs = glob.glob(join(args.input, scene, 'test_preds', '*rgb_cc.jpg'))
        rgbs = np.random.choice(rgbs, args.num_select, replace=False)
        for rgb in rgbs:
            dpt = rgb.replace('_rgb_cc.jpg', '.jpg')
            tar_rgb = join(args.output, f'{scene}_{os.path.basename(rgb)[:-11]}_rgb.jpg')
            tar_dpt = join(args.output, f'{scene}_{os.path.basename(dpt)[:-4]}_depth.jpg')
            os.system(f'ln -s {rgb} {tar_rgb}')
            os.system(f'ln -s {dpt} {tar_dpt}')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/bn/haotongdata/home/linhaotong/projects/zipnerf-pytorch/exp/scannetpp_all_0610')
    parser.add_argument('--output', type=str, default='/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/preview/scannetpp_zipnerf_preview3')
    parser.add_argument('--num_select', type=int, default=5)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
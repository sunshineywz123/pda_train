import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
import imageio
import cv2

from lib.utils.parallel_utils import parallel_execution
sys.path.append('.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--input_dir', type=str, default='datasets/ARKitScenes/download/upsampling/Validation')
    parser.add_argument('--up_scale', type=int, default=8, help='1, 2, 4, 8')
    parser.add_argument('--conf_level', type=int, default=-1, help='-1, 0, 1, 2')
    args = parser.parse_args()
    return args

def evaluate_img(meta):
    input_dir, name, up_scale, conf_level = meta
    
    highres_depth = (imageio.imread(join(input_dir, 'highres_depth', name)) / 1000.)
    lowres_depth = (imageio.imread(join(input_dir, 'lowres_depth', name)) / 1000.)
    
    if up_scale == 1:
        try: conf = imageio.imread(join(input_dir, 'confidence', name)).astype(np.uint8)
        except: conf = np.ones_like(lowres_depth).astype(np.uint8) * 2
        if conf_level == -1: conf_msk = np.ones_like(conf).astype(np.bool_)
        else: conf_msk = conf == conf_level
        highres_depth = cv2.resize(highres_depth, (256, 192), interpolation=cv2.INTER_NEAREST)
    elif up_scale == 2 or up_scale == 4:
        lowres_depth = cv2.resize(lowres_depth, (256*up_scale, 192*up_scale), interpolation=cv2.INTER_LINEAR) 
        highres_depth = cv2.resize(highres_depth, (256*up_scale, 192*up_scale), interpolation=cv2.INTER_NEAREST)
        conf_msk = np.ones_like(lowres_depth).astype(np.bool_)
    elif up_scale == 8:
        lowres_depth = cv2.resize(lowres_depth, (1920, 1440), interpolation=cv2.INTER_LINEAR) 
        conf_msk = np.ones_like(lowres_depth).astype(np.bool_)
        
    msk = highres_depth != 0
    l1 = np.abs(highres_depth - lowres_depth)[msk & conf_msk].mean()
    rmse = np.sqrt(((highres_depth - lowres_depth) ** 2)[msk & conf_msk].mean())
    return [l1, rmse]
    
    

def main(args):
    workspace = os.environ['workspace']
    scenes = sorted(os.listdir(join(workspace, args.input_dir)))
    scenes = scenes[:-1]
    
    metas = []
    for scene in scenes:
        names = os.listdir(join(workspace, args.input_dir, scene, 'wide'))
        metas.extend([(join(workspace, args.input_dir, scene), name, args.up_scale, args.conf_level) for name in names])
    # evaluate_img(metas[0]) 
    results = parallel_execution(
        metas,
        action=evaluate_img,
        print_progress=True,
        num_processes=128,
    )
    results = np.asarray(results)
    print('Evaluation setting: confidence of {}, up_scale of {}'.format(args.conf_level, args.up_scale))
    print('L1 error: ', results[:, 0][~np.isnan(results[:, 0])].mean())
    print('RMSE: ', results[:, 1][~np.isnan(results[:, 1])].mean())

if __name__ == '__main__':
    args = parse_args()
    main(args)
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

min_val, max_val = 0.5, 5.5
num_bin = 10
bins = np.linspace(0.5, 5.5, num_bin + 1)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--input_dir', type=str, default='datasets/ARKitScenes/download/upsampling/Validation')
    parser.add_argument('--up_scale', type=int, default=1, help='1, 2, 4, 8')
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
    l1_diff_map = (lowres_depth - highres_depth)[msk]


    digitized = np.digitize(highres_depth[msk], bins) - 1
    ret = np.zeros((num_bin, 3))
    for i in range(num_bin):
        if np.sum(digitized == i) == 0:
            continue
        ret[i][0] = l1_diff_map[digitized == i].mean()
        ret[i][1] = l1_diff_map[digitized == i].std()
        ret[i][2] = np.sum(digitized == i)
    return ret
    
    

def main(args):
    workspace = os.environ['workspace']
    scenes = sorted(os.listdir(join(workspace, args.input_dir)))
    scenes = scenes[:-1]
    
    metas = []
    for scene in scenes:
        names = os.listdir(join(workspace, args.input_dir, scene, 'wide'))
        metas.extend([(join(workspace, args.input_dir, scene), name, args.up_scale, args.conf_level) for name in names][::5])
    results = parallel_execution(
        metas,
        action=evaluate_img,
        print_progress=True,
        num_processes=128,
    )
    results = np.asarray(results)
    weights = results[:, :, 2]
    weights = weights / weights.sum(axis=0)[None]
    mean = (weights * results[:, :, 0]).sum(axis=0)
    std = np.sqrt((weights * (results[:, :, 1] ** 2)).sum(axis=0))
    print(results[:, :, 2].sum(axis=0)/1000000)
    import matplotlib.pyplot as plt
    plt.plot(bins[:-1], mean, '.-')
    plt.plot(bins[:-1], std, '.-')
    plt.savefig('err_std.jpg', dpi=300)

if __name__ == '__main__':
    args = parse_args()
    main(args)
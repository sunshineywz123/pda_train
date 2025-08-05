import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
import imageio
import open3d as o3d

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dpt_path', type=str)
    args = parser.parse_args()
    return args


def main(args):
    dpt = np.asarray(np.load(args.input_dpt_path)['data'])
    dpt_min, dpt_max = np.percentile(dpt, 2.), np.percentile(dpt, 98.)
    print('Scale:', dpt_max - dpt_min, ' Shift:', dpt_min)
    
if __name__ == '__main__':
    args = parse_args()
    main(args)

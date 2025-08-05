import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')

def main(args):
    pass
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/bn/haotongdata/Datasets/DyDToF')
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)
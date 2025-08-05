import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
import torch
sys.path.append('.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str) # model_path
    parser.add_argument('--remove_prefix', type=str) # model_path
    parser.add_argument('--output', type=str) # out_path
    args = parser.parse_args()
    return args

def main(args):
    model = torch.load(args.input)['state_dict']
    new_model = {}
    for k in model.keys():
        if k[:len(args.remove_prefix)] != args.remove_prefix:
            import ipdb; ipdb.set_trace()
        else:
            new_model[k[len(args.remove_prefix):]] = model[k]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(new_model, args.output)
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
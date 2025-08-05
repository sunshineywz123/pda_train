import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
import cv2
sys.path.append('.')

def main(args):
    input_files = sorted(os.listdir(args.input))
    os.makedirs(args.output, exist_ok=True)
    for input_file in tqdm(input_files):
        depth = np.load(join(args.input, input_file))['data']
        if (depth > 200).any(): 
            import ipdb; ipdb.set_trace()
        depth = (depth * 256).astype(np.uint16)
        output_file = join(args.output, input_file.split('__')[-1].replace('.npz', '.png'))
        cv2.imwrite(output_file, depth)
    os.chdir(os.path.dirname(args.output))
    import ipdb; ipdb.set_trace()
    cmd = 'zip -r %s %s' % ('submit' + '.zip', 'submit' + '/*')
    os.system(cmd)
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/outputs/depth_estimation/aug_hypersim_kitti/results/test_submit/orig_pred')
    parser.add_argument('--output', type=str, default='/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/outputs/depth_estimation/aug_hypersim_kitti/results/test_submit/submit')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)
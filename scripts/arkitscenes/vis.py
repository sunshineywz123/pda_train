from asyncio.constants import LOG_THRESHOLD_FOR_CONNLOST_WRITES
import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
import glob
import imageio
from lib.utils.vis_utils import colorize_depth_maps
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='datasets/ARKitScenes/download/upsampling/Validation')
    parser.add_argument('--output_space', type=str, default='processed_datasets/ARKitScenes/upsampling/Validation')
    args = parser.parse_args()
    return args

def convert_confidence_to_rgb(confidence):
    '''
    confidence should be  hxw map in the int of [0, 1, 2] -> red, green, blue
    '''
    h, w = confidence.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[confidence == 0] = [255, 0, 0]
    rgb[confidence == 1] = [0, 255, 0] 
    rgb[confidence == 2] = [0, 0, 255]
    return rgb
    

def main(args):
    workspace = os.environ.get('workspace')
    frames = [0, 400, 20]
    data_root = join(workspace, args.data_root)
    scenes = sorted(os.listdir(data_root))
    rgb_files = []
    for scene in tqdm(scenes):
        rgb_files += sorted(glob.glob(join(
            data_root, scene, 'wide', '*.png'
        )))
    rgb_files = rgb_files[frames[0]:len(rgb_files):frames[2]]
    
    for rgb_file in tqdm(rgb_files):
        img_name = os.path.basename(rgb_file)
        scene = rgb_file.split('/')[-3]
        
        
        rgb = (np.asarray(imageio.imread(rgb_file))).astype(np.uint8)
        highres_depth = np.asarray(imageio.imread(join(workspace, args.data_root, scene, 'highres_depth', img_name)) / 1000.).astype(np.float32)
        lowres_depth = np.asarray(imageio.imread(join(workspace, args.data_root, scene, 'lowres_depth', img_name)) / 1000.).astype(np.float32)
        dpt_min, dpt_max = lowres_depth.min(), lowres_depth.max()
        confidence = imageio.imread(join(workspace, args.data_root, scene, 'confidence', img_name))
        conf_map = convert_confidence_to_rgb(confidence)
        
        rgb = cv2.resize(rgb, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) 
        highres_depth = cv2.resize(highres_depth, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST) 
        highres_depth_map = (colorize_depth_maps(highres_depth, dpt_min, dpt_max)[0].transpose((1, 2, 0)) * 255.).astype(np.uint8)
        lowres_depth_map = (colorize_depth_maps(lowres_depth, dpt_min, dpt_max)[0].transpose((1, 2, 0)) * 255.).astype(np.uint8)
        h, w = rgb.shape[:2]
        
        gt_lowres_depth = cv2.resize(highres_depth, (256, 192), interpolation=cv2.INTER_NEAREST)
        errmap = np.zeros_like(gt_lowres_depth)
        errmap[gt_lowres_depth != 0] = np.abs(gt_lowres_depth[gt_lowres_depth != 0] - lowres_depth[gt_lowres_depth != 0])
        errmap = (errmap / errmap.max() * 255).astype(np.uint8)
        errmap = cv2.applyColorMap(errmap, cv2.COLORMAP_JET)
        
        
        board = (np.ones((h, w*2+256, 3)) * 255).astype(np.uint8)
        board[:h, :w] = rgb
        board[:h, w:w*2] = highres_depth_map
        board[:192, w*2:] = conf_map
        board[192:192*2, w*2:] = lowres_depth_map
        board[192*2:192*3, w*2:] = errmap
        
        image_path = join(workspace, args.output_space, scene + '_' + img_name)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        imageio.imwrite(image_path, board)
        # print(scene, img_name, confidence.min(), confidence.max())
        # import ipdb; ipdb.set_trace()
        # confidence = imageio.imread(join(workspace, args.data_root, scene, 'confidence', img_name))
        # output_path = join(workspace, args.output_space, rgb_name)
        # imageio.imwrite(output_path, rgb)
    
    
    
    pass

if __name__ == '__main__':
    args = parse_args()
    main(args)
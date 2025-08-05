import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
import imageio
sys.path.append('.')
from lib.utils import vis_utils

def main(args):
    # rgb_dir = '/mnt/data/home/linhaotong/datasets/mycapture/671d34a318/rgb'
    # depth_dir = '/mnt/data/home/linhaotong/datasets/mycapture/671d34a318/depth'
    # tar_dir = 'data/pl_htcode/exp_results/video_depth_results/songyou_iphone_depth_preupsample'
    # read_depth_func = lambda x: np.asarray(imageio.imread(x))/1000
    # video = vis_utils.warp_rgbd_video(rgb_dir, depth_dir, tar_dir, read_depth_func=read_depth_func, frame_sample=[0, 6000, 2], pre_upsample=True)
    
    # rgb_dir = '/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/outputs/depth_estimation/aug_hypersim_arkit_random_all_dataset_zip_grad0.5_new_mask/results/scannetpp_5f99900f09_video_new/vis_depth'
    # depth_dir = '/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/outputs/depth_estimation/aug_hypersim_arkit_random_all_dataset_zip_grad0.5_new_mask/results/scannetpp_5f99900f09_video_new/orig_pred'
    # tar_dir = '/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/outputs/depth_estimation/aug_hypersim_arkit_random_all_dataset_zip_grad0.5_new_mask/results/scannetpp_5f99900f09_video_new/pcd_video'
    # video = vis_utils.warp_rgbd_video(rgb_dir, depth_dir, tar_dir, frame_sample=[0, 600, 1], pre_upsample=True, fps=60, focal=756, depth_format='.npz')
    
    # rgb_dir = '/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/outputs/depth_estimation/aug_hypersim_arkit_random_all_dataset_zip_grad0.5_new_mask/results/scannetpp_5f99900f09_video_new/vis_depth'
    # depth_dir = '/mnt/bn/haotongdata/Datasets/scannetpp/data/5f99900f09/iphone/depth'
    # tar_dir = '/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/outputs/depth_estimation/aug_hypersim_arkit_random_all_dataset_zip_grad0.5_new_mask/results/scannetpp_5f99900f09_video_new/lidar_pcd_video'
    # video = vis_utils.warp_rgbd_video(rgb_dir, depth_dir, tar_dir, frame_sample=[0, 600, 1], pre_upsample=True, fps=60, focal=756, depth_format='.png')
    
    # rgb_dir = '/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/outputs/depth_estimation/aug_hypersim_arkit_random_all_dataset_zip_grad0.5_new_mask/results/scannetpp_5f99900f09_video_new/vis_depth'
    # depth_dir = '/mnt/bn/haotongdata/Datasets/scannetpp/data/5f99900f09/iphone/depth'
    # tar_dir = '/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/outputs/depth_estimation/aug_hypersim_arkit_random_all_dataset_zip_grad0.5_new_mask/results/scannetpp_5f99900f09_video_new/lidar_pcd_video_lowres'
    # video = vis_utils.warp_rgbd_video(rgb_dir, depth_dir, tar_dir, frame_sample=[0, 600, 1], pre_upsample=False, fps=60, focal=756, depth_format='.png')
    
    # rgb_dir = '/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/outputs/depth_estimation/depth_anything_metric/results/scannetpp_5f99900f09_video_new/vis_depth'
    # depth_dir = '/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/outputs/depth_estimation/depth_anything_metric_fit/results/scannetpp_5f99900f09_video_new/orig_pred'
    # tar_dir = '/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/outputs/depth_estimation/depth_anything_metric_fit/results/scannetpp_5f99900f09_video_new/pcd_video'
    # video = vis_utils.warp_rgbd_video(rgb_dir, depth_dir, tar_dir, frame_sample=[0, 600, 1], pre_upsample=False, fps=60, focal=756, depth_format='.npz')
    
    exp_name = 'aug_hypersim_arkit_random_all_dataset_zip_grad0.5_new_mask'
    exp_name = 'depth_anything_metric'
    tag = 'seq4_new'
    rgb_dir = f'/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/outputs/depth_estimation/{exp_name}/results/{tag}/vis_depth'
    depth_dir = f'/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/outputs/depth_estimation/{exp_name}/results/{tag}/orig_pred'
    tar_dir = f'/mnt/bn/haotongdata/home/linhaotong/projects/pl_htcode/data/pl_htcode/outputs/depth_estimation/{exp_name}/results/{tag}/pcd_video'
    video = vis_utils.warp_rgbd_video(rgb_dir, depth_dir, tar_dir, frame_sample=[500, 1100, 1], pre_upsample=True, fps=60, focal=756, depth_format='.npz')

    # rgb_dir = '/mnt/data/home/linhaotong/datasets/mycapture/671d34a318/rgb'
    # depth_dir = '/mnt/data/home/linhaotong/datasets/mycapture/671d34a318/depth'
    # conf_dir = '/mnt/data/home/linhaotong/datasets/mycapture/671d34a318/confidence'
    # tar_dir = 'data/pl_htcode/exp_results/video_depth_results/songyou_iphone_conf_depth'
    # read_depth_func = lambda x: np.asarray(imageio.imread(x))/1000
    # video = vis_utils.warp_rgbd_video(rgb_dir, depth_dir, tar_dir, conf_dir, read_depth_func=read_depth_func, frame_sample=[0, 30000, 2])

    # rgb_dir = '/mnt/data/home/linhaotong/datasets/mycapture/671d34a318/rgb'
    # depth_dir = '/mnt/data/home/linhaotong/projects/pl_htcode/data/pl_htcode/outputs/depth_estimation/july_hypersim_baseline0717_minmax_new_fixbug_1nodes/results/default/orig_pred'
    # tar_dir = 'data/pl_htcode/exp_results/video_depth_results/songyou_iphone_depth_minmax'
    # video = vis_utils.warp_rgbd_video(rgb_dir, depth_dir, tar_dir, frame_sample=[0, 6000, 2], depth_format='.npz')
    
    # rgb_dir = '/mnt/data/home/linhaotong/datasets/mycapture/671d34a318/rgb'
    # depth_dir = '/mnt/data/home/linhaotong/projects/pl_htcode/data/pl_htcode/outputs/depth_estimation/repeat_aug_zipmesh_arkit/results/default/orig_pred'
    # tar_dir = 'data/pl_htcode/exp_results/video_depth_results/songyou_iphone_depth_aug_zipmesh_arkit'
    # video = vis_utils.warp_rgbd_video(rgb_dir, depth_dir, tar_dir, frame_sample=[0, 6000, 2], depth_format='.npz')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb_dir', type=str)
    parser.add_argument('--depth_dir', type=str)
    parser.add_argument('--tar_dir', type=str)
    parser.add_argument('--frame_start', type=int, default=0)
    parser.add_argument('--frame_end', type=int, default=600)
    parser.add_argument('--frame_interval', type=int, default=1)
    parser.add_argument('--pre_upsample', type=int, default=0)
    parser.add_argument('--fps', type=int, default=60)
    parser.add_argument('--focal', type=int, default=756)
    parser.add_argument('--tar_h', type=int, default=756)
    parser.add_argument('--tar_w', type=int, default=1008)
    parser.add_argument('--depth_format', type=str, default='.npz')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    video = vis_utils.warp_rgbd_video(args.rgb_dir, 
                                      args.depth_dir, 
                                      args.tar_dir, 
                                      frame_sample=[args.frame_start, args.frame_end, args.frame_interval], 
                                      pre_upsample=args.pre_upsample, 
                                      fps=args.fps, 
                                      focal=args.focal, 
                                      depth_format=args.depth_format,
                                      tar_h=args.tar_h,
                                      tar_w=args.tar_w)
    # main(args)
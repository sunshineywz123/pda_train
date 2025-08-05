import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
sys.path.append('./trdparties/DepthAnything')
import cv2
import imageio
import re

from trdparties.DepthAnything.depth_anything.dpt import DepthAnything
from trdparties.DepthAnything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import torch
from torchvision.transforms import Compose
import torch.nn.functional as F

def cluster_indexes(nums, thresh):
    """
    This function takes a sorted array of numbers (in descending order) and a threshold value.
    It groups the numbers into clusters based on their indexes, where the difference between the 
    maximum and minimum numbers in each cluster does not exceed the threshold.
    
    Parameters:
    - nums: List[float], a sorted list of numbers in descending order.
    - thresh: float, the maximum allowed difference between the maximum and minimum numbers in each cluster.
    
    Returns:
    - List[List[int]], a list of clusters, each cluster is a list of indexes.
    """
    
    clusters = []  # Initialize an empty list to store the clusters
    # if nums:
    current_cluster_indexes = [0]  # Initialize the first cluster's indexes
    
    for i in range(1, len(nums)):
        # Check if the current number can be added to the existing cluster based on threshold
        if nums[current_cluster_indexes[0]] - nums[i] <= thresh:
            current_cluster_indexes.append(i)
        else:
            # If the current number cannot be added to the existing cluster, start a new cluster of indexes
            clusters.append(current_cluster_indexes)
            current_cluster_indexes = [i]
    
    # Add the last cluster of indexes if it exists
    if current_cluster_indexes:
        clusters.append(current_cluster_indexes)
    
    return clusters
    
def get_color(item, weight_thresh, disp_thresh):
    if len(item) == 0: return np.zeros(3), False
    weight = np.sum([i[1] for i in item])
    if weight < weight_thresh: return np.zeros(3), False
    
    disps = [i[0] for i in item]
    colors = [i[2] for i in item]
    weights = [i[1] for i in item]
    
    # sorted by disp
    idx = np.argsort(disps)[::-1]
    disps = np.array(disps)[idx]
    colors = np.array(colors)[idx]
    weights = np.array(weights)[idx]
    
    clusters = cluster_indexes(disps, disp_thresh)
    
    cluster_weights = [np.sum(weights[cluster]) for cluster in clusters]
    cluster_colors = [np.sum(colors[cluster] * weights[cluster][..., None], axis=0) / cluster_weights[i] for i, cluster in enumerate(clusters)]
    
    for i in range(len(cluster_weights)):
        if cluster_weights[i] > weight_thresh:
            return cluster_colors[i], True
    return np.zeros(3), False


def add_item(img_right, disp, weight, color, i, j):
    img_right[i, j].append((disp, weight, color))

def create_array_with_empty_lists(h, w):
    # 创建一个指定形状的空对象数组
    arr = np.empty((h, w), dtype=object)
    # 填充每个元素为空列表
    for i in range(h):
        for j in range(w):
            arr[i, j] = []
    return arr

def warp_image_fine_sort_onlyx(img, disp, weight_thresh=0.01, disp_thresh=2.0):
    # 假设每一个pixel contribute 到右图2个pixels
    # 最后做normalize
    msk = (~np.isinf(disp)) 
    img_right = np.zeros_like(img)
    msk_right = np.zeros_like(img[..., 0])
    h, w, _ = img.shape
    img_right = create_array_with_empty_lists(h, w)
    
    for i in range(h):
        for j in range(w):
            if msk[i,j]:
                new_j = j - disp[i,j]
                new_j_round = round(new_j)
                if new_j_round > new_j: new_j_round -= 1
                new_j_ceil = new_j_round + 1
                weight = (new_j_ceil - new_j) 
                if new_j_round >= 0 and new_j_round < w: add_item(img_right, disp[i, j], weight, img[i, j], i, new_j_round)
                weight = (new_j - new_j_round)
                if new_j_ceil >= 0 and new_j_ceil < w: add_item(img_right, disp[i, j], weight, img[i, j], i, new_j_ceil)
    
    valid_msk = np.zeros_like(img[..., 0])
    ret_img = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            ret_img[i, j], valid_msk[i, j] = get_color(img_right[i, j], weight_thresh, disp_thresh)
    return ret_img, valid_msk


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
encoder = 'vitl'
depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(DEVICE).eval()
def predict_depth(image, model, use_disp=True):
    # raw_image = cv2.imread(img_path)
    # image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    h, w = image.shape[:2]
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
        
    with torch.no_grad():
        depth = model(image)
    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth_numpy = depth.detach().cpu().numpy()
    if not use_disp:
        depth_numpy  = np.clip(depth_numpy, 0.01, None)
        depth_numpy = 1 / depth_numpy
    return depth_numpy



def recover_align_dpt(gt, dpt, msk):
    gt = gt[msk]
    pred = dpt[msk]
    # valid_mask = create_valid_mask(sdpt, h, w, dpt_h, dpt_w)
    # pred = dpt[valid_mask]
    a, b = np.polyfit(pred, gt, deg=1)
        
    if a > 0:
        pred_metric = a * dpt + b
    else:
        pred_mean = np.mean(pred)
        gt_mean = np.mean(gt)
        pred_metric = dpt * (gt_mean / pred_mean)
    return pred_metric


def warp_image(img, disp=None, disp_align=None, weight_thresh=0.01, disp_thresh=2.0, baseline=0.06):
    if disp is not None: return warp_image_fine_sort_onlyx(img, disp, weight_thresh, disp_thresh)
    
    disp = predict_depth(img, depth_anything)
    # disp = depth anything
    if disp_align is not None:
        disp_pred = recover_align_dpt(disp_align, disp, ~np.isinf(disp_align))
        return warp_image_fine_sort_onlyx(img, disp_pred, weight_thresh, disp_thresh)
    
        
    # align disp to 1/zoe depth    

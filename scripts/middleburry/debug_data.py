import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
import cv2
import imageio
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/mnt/data/home/linhaotong/projects/pl_htcode/data/pl_htcode/middlebury/MiddEval3/trainingQ')
    parser.add_argument('--output_dir', type=str, default='/nas/home/linhaotong/middlebury')
    parser.add_argument('--weight_thresh', type=float, default=0.01)
    parser.add_argument('--disp_thresh', type=float, default=2)
    args = parser.parse_args()
    return args


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def warp_image_coarse(img_left, disp_left):
    msk = ~np.isinf(disp_left) 
    img_right = np.zeros_like(img_left)
    msk_right = np.zeros_like(img_left[..., 0])
    for i in range(img_left.shape[0]):
        for j in range(img_left.shape[1]):
            if msk[i,j]:
                new_j = int(j - disp_left[i,j])
                if new_j >= 0 and new_j < img_left.shape[1]:
                    img_right[i, new_j] = img_left[i,j]
                    msk_right[i, new_j] += 1
    return img_right, msk_right


def warp_image_fine(img, disp, dispy):
    # 假设每一个pixel contribute 到右图4个pixels
    # 最后做normalize
    msk = (~np.isinf(disp)) & (~np.isinf(dispy))
    img_right = np.zeros_like(img)
    msk_right = np.zeros_like(img[..., 0])
    h, w, _ = img.shape
    for i in range(h):
        for j in range(w):
            if msk[i,j]:
                new_j = j - disp[i,j]
                new_i = i - dispy[i,j]
                # contribute to 4 pixels according linear weight
                new_j_round = round(new_j)
                if new_j_round > new_j: new_j_round -= 1
                new_i_round = round(new_i)
                if new_i_round > new_i: new_i_round -= 1
                new_j_ceil = new_j_round + 1
                new_i_ceil = new_i_round + 1
                # pixel 1
                weight = (new_j_ceil - new_j) * (new_i_ceil - new_i)
                if new_j_round >= 0 and new_j_round < w and new_i_round >= 0 and new_i_round < h:
                    img_right[new_i_round, new_j_round] += img[i,j] * weight
                    msk_right[new_i_round, new_j_round] += weight
                # pixel 2
                weight = (new_j - new_j_round) * (new_i_ceil - new_i)
                if new_j_ceil >= 0 and new_j_ceil < w and new_i_round >= 0 and new_i_round < h:
                    img_right[new_i_round, new_j_ceil] += img[i,j] * weight
                    msk_right[new_i_round, new_j_ceil] += weight
                # pixel 3
                weight = (new_j_ceil - new_j) * (new_i - new_i_round)
                if new_j_round >= 0 and new_j_round < w and new_i_ceil >= 0 and new_i_ceil < h:
                    img_right[new_i_ceil, new_j_round] += img[i,j] * weight
                    msk_right[new_i_ceil, new_j_round] += weight
                # pixel 4
                weight = (new_j - new_j_round) * (new_i - new_i_round)
                if new_j_ceil >= 0 and new_j_ceil < w and new_i_ceil >= 0 and new_i_ceil < h:
                    img_right[new_i_ceil, new_j_ceil] += img[i,j] * weight
                    msk_right[new_i_ceil, new_j_ceil] += weight
    valid_msk = msk_right > 0.01
    img_right[~valid_msk] = 0.
    img_right[valid_msk] /= msk_right[valid_msk][..., None]
    return img_right, valid_msk


def add_item(img_right, disp, weight, color, i, j):
    img_right[i, j].append((disp, weight, color))
    
    
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
    
def create_array_with_empty_lists(h, w):
    # 创建一个指定形状的空对象数组
    arr = np.empty((h, w), dtype=object)
    # 填充每个元素为空列表
    for i in range(h):
        for j in range(w):
            arr[i, j] = []
    return arr

def warp_image_fine_sort(img, disp, dispy, args):
    # 假设每一个pixel contribute 到右图4个pixels
    # 最后做normalize
    msk = (~np.isinf(disp)) & (~np.isinf(dispy))
    img_right = np.zeros_like(img)
    msk_right = np.zeros_like(img[..., 0])
    h, w, _ = img.shape
    img_right = create_array_with_empty_lists(h, w)
    
    for i in range(h):
        for j in range(w):
            if msk[i,j]:
                new_j = j - disp[i,j]
                new_i = i - dispy[i,j]
                # contribute to 4 pixels according linear weight
                new_j_round = round(new_j)
                if new_j_round > new_j: new_j_round -= 1
                new_i_round = round(new_i)
                if new_i_round > new_i: new_i_round -= 1
                new_j_ceil = new_j_round + 1
                new_i_ceil = new_i_round + 1
                # pixel 1
                weight = (new_j_ceil - new_j) * (new_i_ceil - new_i)
                if new_j_round >= 0 and new_j_round < w and new_i_round >= 0 and new_i_round < h:
                    add_item(img_right, disp[i, j], weight, img[i, j], new_i_round, new_j_round)
                    # img_right[new_i_round, new_j_round] += img[i,j] * weight
                    # msk_right[new_i_round, new_j_round] += weight
                # pixel 2
                weight = (new_j - new_j_round) * (new_i_ceil - new_i)
                if new_j_ceil >= 0 and new_j_ceil < w and new_i_round >= 0 and new_i_round < h:
                    add_item(img_right, disp[i, j], weight, img[i, j], new_i_round, new_j_ceil)
                    # img_right[new_i_round, new_j_ceil] += img[i,j] * weight
                    # msk_right[new_i_round, new_j_ceil] += weight
                # pixel 3
                weight = (new_j_ceil - new_j) * (new_i - new_i_round)
                if new_j_round >= 0 and new_j_round < w and new_i_ceil >= 0 and new_i_ceil < h:
                    add_item(img_right, disp[i, j], weight, img[i, j], new_i_ceil, new_j_round)
                    # img_right[new_i_ceil, new_j_round] += img[i,j] * weight
                    # msk_right[new_i_ceil, new_j_round] += weight
                # pixel 4
                weight = (new_j - new_j_round) * (new_i - new_i_round)
                if new_j_ceil >= 0 and new_j_ceil < w and new_i_ceil >= 0 and new_i_ceil < h:
                    add_item(img_right, disp[i, j], weight, img[i, j], new_i_ceil, new_j_ceil)
                    # img_right[new_i_ceil, new_j_ceil] += img[i,j] * weight
                    # msk_right[new_i_ceil, new_j_ceil] += weight
    
    valid_msk = np.zeros_like(img[..., 0])
    ret_img = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            ret_img[i, j], valid_msk[i, j] = get_color(img_right[i, j], args.weight_thresh, args.disp_thresh)
    return ret_img, valid_msk



def warp_image_fine_sort_onlyx(img, disp, args):
    # 假设每一个pixel contribute 到右图4个pixels
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
                # contribute to 4 pixels according linear weight
                new_j_round = round(new_j)
                if new_j_round > new_j: new_j_round -= 1
                new_j_ceil = new_j_round + 1
                # pixel 1
                weight = (new_j_ceil - new_j) 
                if new_j_round >= 0 and new_j_round < w:
                    add_item(img_right, disp[i, j], weight, img[i, j], i, new_j_round)
                    # add_item(img_right, disp[i, j], weight, img[i, j], new_i_round, new_j_round)
                    # img_right[new_i_round, new_j_round] += img[i,j] * weight
                    # msk_right[new_i_round, new_j_round] += weight
                # pixel 2
                weight = (new_j - new_j_round)
                if new_j_ceil >= 0 and new_j_ceil < w:
                    add_item(img_right, disp[i, j], weight, img[i, j], i, new_j_ceil)
                    # img_right[new_i_round, new_j_ceil] += img[i,j] * weight
                    # msk_right[new_i_round, new_j_ceil] += weight
    
    valid_msk = np.zeros_like(img[..., 0])
    ret_img = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            ret_img[i, j], valid_msk[i, j] = get_color(img_right[i, j], args.weight_thresh, args.disp_thresh)
    return ret_img, valid_msk


def process_one_scene(input_dir, scene, output_dir, args):
    input_path = join(input_dir, scene)
    img_left = (imageio.imread(join(input_dir, scene, 'im0.png')) / 255.).astype(np.float32)
    disp_left, disp_left_scale = read_pfm(join(input_dir, scene, 'disp0GT.pfm'))
    
    if os.path.exists(join(input_dir, scene, 'disp0GTy.pfm')):
        dispy_left, dispy_left_scale = read_pfm(join(input_dir, scene, 'disp0GTy.pfm'))
    else:
        dispy_left = None
    if dispy_left is not None:
        warp_img_right, msk_right = warp_image_fine_sort(img_left, disp_left, dispy_left, args)
    else:
        warp_img_right, msk_right = warp_image_fine_sort_onlyx(img_left, disp_left, args)
    
    os.makedirs(join(output_dir, scene), exist_ok=True)
    os.system('cp {} {}'.format(join(input_path, 'im0.png'), join(output_dir, scene, 'im0.png')))
    os.system('cp {} {}'.format(join(input_path, 'im1.png'), join(output_dir, scene, 'im1.png')))
    imageio.imwrite(join(output_dir, scene, 'warp_im1_gtdisp.png'), (np.clip(warp_img_right, 0., 1.) * 255).astype(np.uint8))
    imageio.imwrite(join(output_dir, scene, 'warp_im1_gtdisp_mask.png'), msk_right.astype(np.uint8) * 255)
    

def main(args):
    for scene in tqdm(os.listdir(args.input_dir)):
        process_one_scene(args.input_dir, scene, args.output_dir, args)

if __name__ == '__main__':
    args = parse_args()
    main(args)
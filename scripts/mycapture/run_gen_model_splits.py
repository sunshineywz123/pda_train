import os
from os.path import join
import argparse
import sys
import ipdb
import numpy as np
import cv2
from tqdm import tqdm
import shutil
import sys
sys.path.append('.')
from lib.utils.parallel_utils import parallel_execution
import imageio

def variance_of_laplacian(image):
    """Compute the variance of the Laplacian of the image."""
    return cv2.Laplacian(image, cv2.CV_64F).var()

def calculate_sharpness(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is not None:
        sharpness = variance_of_laplacian(image)
        return (os.path.basename(image_path), sharpness)
    return (os.path.basename(image_path), None)

def detect_blurry_frames(input_dir, output_clear_dir, top_percent=30, max_interval=120, min_interval=6, soft_link=True, max_number=500):
    
    os.system('rm -rf ' + output_clear_dir)
    os.makedirs(output_clear_dir, exist_ok=True)

    sharpness_values = []
    image_files = sorted([f for f in os.listdir(input_dir) if (f.endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.'))])
    image_paths = [join(input_dir, f) for f in image_files]
    
    sharpness_values = parallel_execution(
        image_paths,
        action=calculate_sharpness,
        print_progress=True,
        desc="Calculating sharpness"
    )

    sharpness_values.sort(key=lambda x: x[1], reverse=True)  # Sort by sharpness in descending order
    num_to_select = min(max(1, len(sharpness_values) // top_percent), max_number)
    # 最多选最好的500张
    selected_frames = []
    selected_frames_ids = []
    # last_selected_frame_index = -5  # Initialize to ensure first frame can be selected

    # Select sharp frames ensuring the interval is at least 5 frames
    # for i in range(len(sharpness_values)):
    #     frame_index = image_files.index(sharpness_values[i][0])
    #     if frame_index >= last_selected_frame_index + 5:
    #         selected_frames.append(sharpness_values[i][0])
    #         last_selected_frame_index = frame_index
    #         if len(selected_frames) >= num_to_select:
    #             break
    image_file_index_map = {name: idx for idx, name in enumerate(image_files)}
    progress_bar = tqdm(range(len(sharpness_values)), desc='Selecting sharp frames')
    for i in progress_bar:
        frame_name = sharpness_values[i][0]
        frame_index = image_file_index_map[frame_name]
        progress_bar.set_postfix({
            'Selected Frames': len(selected_frames)
        })
        
        if len(selected_frames) == 0:
            selected_frames.append(frame_name)
            selected_frames_ids.append(frame_index)
            continue
        
        if np.abs(np.asarray(selected_frames_ids) - frame_index).min() >= min_interval:
            selected_frames.append(frame_name)
            selected_frames_ids.append(frame_index)
            
            if len(selected_frames) >= num_to_select:
                break

    # Sliding window to ensure at least one frame per second (60 frames) is selected
    final_selected_frames = set(selected_frames)
    for start in tqdm(range(0, len(image_files), max_interval), desc='Sliding window'):
        window_frames = image_files[start:start+max_interval]
        # if not any(frame in final_selected_frames for frame in window_frames):
        if not final_selected_frames.intersection(window_frames):
            best_frame_in_window = max(window_frames, key=lambda f: dict(sharpness_values)[f])
            final_selected_frames.add(best_frame_in_window)

    # Copy selected and non-selected frames to corresponding directories
    print('Num of selected frames:', len(final_selected_frames))
    print('Num of total frames:', len(image_files))
    for image_file in tqdm(image_files, desc="Sorting images"):
        src_path = join(input_dir, image_file)
        if image_file in final_selected_frames:
            dst_path = join(output_clear_dir, image_file)
            if soft_link:
                os.system('ln -s ' + src_path + ' ' + dst_path)
            else:
                shutil.copy(src_path, dst_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data')
    parser.add_argument('--scene', type=str, default='5f99900f09')
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--start_number', type=int, default=0)
    parser.add_argument('--end_number', type=int, default=600)
    parser.add_argument('--interval', type=int, default=5)
    args = parser.parse_args()
    return args

def main(args):
    data_root = join(args.input_data_dir, args.scene, 'iphone')
    output_dir = join(data_root, 'split_output/split_{}'.format(args.split))
    os.makedirs(output_dir, exist_ok=True)
    # image, depth, confidence
    rgbs = sorted([f for f in os.listdir(join(data_root, 'rgb')) if f.endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.')])
    depths = sorted([f for f in os.listdir(join(data_root, 'depth')) if f.endswith('.png') and not f.startswith('.')])
    if os.path.exists(join(data_root, 'confidence')):
        confidences = sorted([f for f in os.listdir(join(data_root, 'confidence')) if f.endswith('.png') and not f.startswith('.')])
    else:
        confidences = None
        
    os.makedirs(join(output_dir, 'images'), exist_ok=True)
    os.makedirs(join(output_dir, 'depth'), exist_ok=True)
    if confidences is not None: os.makedirs(join(output_dir, 'confidence'), exist_ok=True)
    for i in range(args.start_number, args.end_number, args.interval):
        rgb = rgbs[i]
        depth = depths[i]
        if confidences: confidence = confidences[i]
        else: confidence = None
        
        tar_rgb = join(output_dir, 'images', rgb)
        os.system('ln -s {} {}'.format(join(data_root, 'rgb', rgb), tar_rgb))
        
        
        tar_depth = join(output_dir, 'depth', depth)
        os.system('ln -s {} {}'.format(join(data_root, 'depth', depth), tar_depth))
        
        if confidence is not None:
            tar_confidence = join(output_dir, 'confidence', confidence)
            os.system('ln -s {} {}'.format(join(data_root, 'confidence', confidence), tar_confidence))
    
    
    
if __name__ == '__main__':
    args = parse_args()
    main(args)

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

def variance_of_laplacian(image):
    """Compute the variance of the Laplacian of the image."""
    return cv2.Laplacian(image, cv2.CV_64F).var()

def calculate_sharpness(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is not None:
        sharpness = variance_of_laplacian(image)
        return (os.path.basename(image_path), sharpness)
    return (os.path.basename(image_path), None)

def detect_blurry_frames(input_dir, output_clear_dir, top_percent=30, max_interval=120, min_interval=6):
    
    output_mask_dir = output_clear_dir.replace('images', 'masks')
    os.system('rm -rf ' + output_clear_dir)
    os.system('rm -rf ' + output_mask_dir)
    # os.system('rm -rf ' + output_blurry_dir)
    os.makedirs(output_clear_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    # if not os.path.exists(output_blurry_dir):
        # os.makedirs(output_blurry_dir)

    sharpness_values = []
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    image_paths = [join(input_dir, f) for f in image_files]
    
    # Calculate sharpness for each image
    sharpness_values = parallel_execution(
        image_paths,
        action=calculate_sharpness,
        print_progress=True,
        desc="Calculating sharpness"
    )
    # for image_file in tqdm(image_files, desc="Calculating sharpness"):
    #     image_path = join(input_dir, image_file)
    #     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #     if image is not None:
    #         sharpness = variance_of_laplacian(image)
    #         sharpness_values.append((image_file, sharpness))

    sharpness_values.sort(key=lambda x: x[1], reverse=True)  # Sort by sharpness in descending order
    num_to_select = min(max(1, len(sharpness_values) // top_percent), 500)
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
            os.system('ln -s ' + src_path + ' ' + dst_path)
            src_path = join(input_dir.replace('rgb', 'rgb_masks'), image_file.replace('.jpg', '.png'))
            dst_path = join(output_mask_dir, image_file + '.png')
            os.system('ln -s ' + src_path + ' ' + dst_path)
            # shutil.copy(src_path, dst_path)
        # else:
        #     dst_path = join(output_blurry_dir, image_file)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_rgb_dir', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/98b4ec142f/iphone/rgb')
    parser.add_argument('--output_clear_rgb_dir', type=str, default='/mnt/bn/haotongdata/Datasets/scannetpp/data/98b4ec142f/merge_dslr_iphone/images/iphone')
    parser.add_argument('--scene', type=str, default=None)
    args = parser.parse_args()
    return args

def main(args):
    input_rgb_dir = args.input_rgb_dir
    clear_rgb_dir = args.output_clear_rgb_dir
    if args.scene is not None:
        input_rgb_dir = input_rgb_dir.replace('98b4ec142f', args.scene)
        clear_rgb_dir = clear_rgb_dir.replace('98b4ec142f', args.scene)
    detect_blurry_frames(input_rgb_dir, clear_rgb_dir)

if __name__ == '__main__':
    args = parse_args()
    main(args)

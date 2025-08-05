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
import zlib
import lz4.block
import imageio as iio

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


def extract_depth(depth_bin, depth_dir, sample_rate=1):
    os.makedirs(depth_dir, exist_ok=True)
    height, width = 192, 256
    frame_id = 0
    with open(depth_bin, 'rb') as infile:
        while True:
            size = infile.read(4)   # 32-bit integer
            if len(size) == 0:
                break
            size = int.from_bytes(size, byteorder='little')
            if frame_id % sample_rate != 0:
                infile.seek(size, 1)
                frame_id += 1
                continue
            data = infile.read(size)
            try:
                # try using lz4
                data = lz4.block.decompress(data, uncompressed_size=height * width * 2)  # UInt16 = 2bytes
                depth = np.frombuffer(data, dtype=np.uint16).reshape(height, width)
            except:
                # try using zlib
                data = zlib.decompress(data, wbits=-zlib.MAX_WBITS)
                depth = np.frombuffer(data, dtype=np.float32).reshape(height, width)
                depth = (depth * 1000).astype(np.uint16)
            # 6 digit frame id = 277 minute video at 60 fps
            iio.imwrite(f"{depth_dir}/{frame_id:06}.png", depth)
            frame_id += 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', type=str, default='/mnt/bn/haotongdata/Datasets/mycapture_arkit')
    parser.add_argument('--scene', type=str, default='20240522_bicycle')
    parser.add_argument('--top_percent', type=int, default=15)
    parser.add_argument('--max_interval', type=int, default=60)
    parser.add_argument('--min_interval', type=int, default=6)
    args = parser.parse_args()
    return args

def main(args):
    data_root = join(args.input_data_dir, args.scene)
    # os.makedirs(join(data_root, 'raw'), exist_ok=True)
    # os.system('mv {}/*.* {}/raw'.format(data_root, data_root))
    # if not os.path.exists(join(data_root, 'rgb')):
    #     os.makedirs(join(data_root, 'rgb'), exist_ok=True)
    #     os.system(f'ffmpeg -i {join(data_root, "raw/rgb.mp4")} -start_number 0 -q:v 8 {join(data_root, "rgb")}/%06d.jpg')
    # detect_blurry_frames(join(data_root, 'rgb'), join(data_root, 'images'), top_percent=args.top_percent, max_interval=args.max_interval, min_interval=args.min_interval)

    depth_bin = join(data_root, 'raw/depth.bin')
    depth_dir = join(data_root, 'depth')
    extract_depth(depth_bin, depth_dir)

if __name__ == '__main__':
    args = parse_args()
    main(args)

import os
from os.path import join
import argparse
import sys
import numpy as np
from requests import get
from tqdm import tqdm
import cv2
sys.path.append('.')
from lib.utils.pylogger import Log
from lib.utils.vis_utils import merge
import json

def get_img(cap, frame_number, total_frame=None):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    if total_frame is None:
        total_frame =  int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read()
    if not ret:
       Log.error("Failed to retrieve frame")
       raise Exception("Failed to retrieve frame")
    text = f"Frame: {frame_number}/{total_frame}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

def get_cur_frame(caps, frames, total_frames):
    return [get_img(caps[i], frames[i], total_frames[i]) for i in range(len(caps))]
    
selected_video = 0
output_cnt = 0
ranges = None
cur_frames = None
output_json = None

def main(args):
    video_paths = ['/Users/linhaotong/Downloads/30/seq1/rgb.mp4',
                   '/Users/linhaotong/Downloads/31/seq1/rgb.mp4',
                   '/Users/linhaotong/Downloads/32/seq1/rgb.mp4',
                   '/Users/linhaotong/Downloads/33/seq1/rgb.mp4',
                   '/Users/linhaotong/Downloads/34/seq1/rgb.mp4']
    global output_json
    output_json = '/Users/linhaotong/Downloads'
    caps = [cv2.VideoCapture(video_path) for video_path in video_paths]
    total_frames = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
    global cur_frames
    cur_frames = [0 for cap in caps]
    cur_frame = get_cur_frame(caps, cur_frames, total_frames)
    global ranges
    img, ranges = merge(cur_frame, resize=True, ret_range=True, resize_height=1440)


    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            global selected_video
            global ranges
            for i, (x1, y1, x2, y2) in enumerate(ranges):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    selected_video = i
                    break
    cv2.namedWindow("Video Frames")
    cv2.setMouseCallback("Video Frames", click_event)

    while True:
        cur_frame = get_cur_frame(caps, cur_frames, total_frames)
        cur_img = merge(cur_frame, resize=True, ret_range=False, resize_height=1440)
        cv2.imshow("Video Frames", cur_img)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == 81 or key == ord('a'):  # Left arrow or 'a'
            cur_frames[selected_video] = max(0, cur_frames[selected_video] - 1)
        elif key == 82 or key == ord('w'):  # Up arrow or 'w'
            cur_frames[selected_video] = max(0, cur_frames[selected_video] - 10)
        elif key == ord('1'):  # Right arrow or 'd'
            cur_frames[selected_video] = max(0, cur_frames[selected_video] - 100)
        elif key == 83 or key == ord('d'):  # Right arrow or 'd'
            cur_frames[selected_video] = min(total_frames[selected_video] - 2, cur_frames[selected_video] + 1)
        elif key == 84 or key == ord('s'):  # Down arrow or 's'
            cur_frames[selected_video] = min(total_frames[selected_video] - 2, cur_frames[selected_video] + 10)
        elif key == ord('2'):  # Right arrow or 'd'
            cur_frames[selected_video] = min(total_frames[selected_video] - 2, cur_frames[selected_video] + 100)
        elif key == ord('2'):  # Right arrow or 'd'
            cur_frames[selected_video] = min(total_frames[selected_video] - 2, cur_frames[selected_video] + 100)
        elif key == ord('f'):  # Down arrow or 's'
            global output_cnt
            output_path = join(output_json, f'syn_{output_cnt}.json')
            json.dump({
                'video_path': video_paths,
                'frame_number': cur_frames
            }, open(output_path, 'w'))
            output_cnt += 1
            Log.info(f"Saved to {output_path}")
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)
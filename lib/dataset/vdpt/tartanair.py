import os
import numpy as np
from tqdm.auto import tqdm
from lib.dataset.vdpt.base_dataset import Dataset as BaseDataset
from copy import deepcopy
import imageio
import glob
import cv2
import json
from lib.utils.pylogger import Log

class Dataset(BaseDataset):
    def build_metas(self):
        root_dir = '/'.join(self.cfg.data_root.split('/')[:-1])
        if os.path.exists(os.path.join(root_dir, 'metas.json')) \
            and os.path.exists(os.path.join(root_dir, 'rgb_files.json')) \
            and os.path.exists(os.path.join(root_dir, 'dpt_files.json')) \
            and self.cfg.get('use_cache', True):
                rgb_files = json.load(open(os.path.join(root_dir, 'rgb_files.json'), 'r'))
                dpt_files = json.load(open(os.path.join(root_dir, 'dpt_files.json'), 'r'))
                global_metas = json.load(open(os.path.join(root_dir, 'metas.json'), 'r'))
        else:
            scenes = sorted(os.listdir(self.cfg.data_root))
            rgb_files = []
            dpt_files = []
            global_metas = []
            for scene in tqdm(scenes):
                for type in ['Easy', 'Hard']:
                    if not os.path.exists(os.path.join(self.cfg.data_root, scene, type)):
                        break
                    seqs = sorted(os.listdir(os.path.join(self.cfg.data_root, scene, type)))
                    for seq in seqs:
                        rgb_video = sorted(glob.glob(os.path.join(self.cfg.data_root, scene, type, seq, 'image_left', '*.png')))
                        dpt_video = sorted(glob.glob(os.path.join(self.cfg.data_root, scene, type, seq, 'depth_left', '*.npy')))
                        video_len = len(rgb_video)
                        metas = self.get_metas_from_videos(len(rgb_video), len(rgb_files), self.cfg.get('frame_len', 8))
                        rgb_files.extend(rgb_video)
                        dpt_files.extend(dpt_video)
                        global_metas.extend(metas)
                        
                        rgb_video = sorted(glob.glob(os.path.join(self.cfg.data_root, scene, type, seq, 'image_left', '*.png')))
                        dpt_video = sorted(glob.glob(os.path.join(self.cfg.data_root, scene, type, seq, 'depth_left', '*.npy')))
                        metas = self.get_metas_from_videos(len(rgb_video), len(rgb_files), self.cfg.get('frame_len', 8))
                        rgb_files.extend(rgb_video)
                        dpt_files.extend(dpt_video)
                        global_metas.extend(metas)
            json_data = json.dumps(rgb_files, ensure_ascii=False, indent=4)
            with open(os.path.join(root_dir, 'rgb_files.json'), 'w', encoding='utf-8') as file:
                file.write(json_data)
            json_data = json.dumps(dpt_files, ensure_ascii=False, indent=4)
            with open(os.path.join(root_dir, 'dpt_files.json'), 'w', encoding='utf-8') as file:
                file.write(json_data)
            converted_metas = [(int(item[0]), [int(x) for x in item[1]]) for item in global_metas]
            json_data = json.dumps(converted_metas, indent=4)
            with open(os.path.join(root_dir, 'metas.json'), 'w', encoding='utf-8') as file:
                file.write(json_data)
        assert(len(rgb_files) == len(dpt_files)), f'{len(rgb_files)} != {len(dpt_files)}'
        frame_len = len(global_metas)
        frame_sample = deepcopy(self.cfg.frames)
        frame_sample[1] =  frame_len if frame_sample[1] == -1 else frame_sample[1]
        # rgb_files = rgb_files[frame_sample[0]:frame_sample[1]:frame_sample[2]]
        # dpt_files = dpt_files[frame_sample[0]:frame_sample[1]:frame_sample[2]]
        global_metas = global_metas[frame_sample[0]:frame_sample[1]:frame_sample[2]]
        self.rgb_files = rgb_files
        self.dpt_files = dpt_files
        self.metas = global_metas
        Log.info(f'[Dataset]: {len(self.metas)} frames in total.')
        
    def read_rgb(self, index): 
        rgb_path = self.rgb_files[index]
        rgb = (np.asarray(imageio.imread(rgb_path)) / 255.).astype(np.float32)
        return rgb
    
    def read_rgbs(self, index):
        frame_idx, seq_ids = self.metas[index]
        rgbs = np.asarray([self.read_rgb(seq_id) for seq_id in seq_ids])
        return rgbs
    
    def read_dpts(self, index):
        frame_idx, seq_ids = self.metas[index]
        dpts = np.asarray([self.read_dpt(seq_id) for seq_id in seq_ids])
        if self.cfg.get('depth_norm', 'simp') == 'simp':
            dpts = np.clip(dpts, 0.5, 80.)
            dpt_min, dpt_max = np.percentile(dpts, 2.), np.percentile(dpts, 98.)
            if dpt_max - dpt_min < 1e-6: dpt_max = dpt_min + 2e-6
            dpts = np.clip(dpts, dpt_min, dpt_max)
            dpts = (dpts - dpt_min) / (dpt_max - dpt_min)
        elif self.cfg.get('depth_norm', 'simp') == 'disp':
            dpts = np.clip(dpts, 1., None)
            dpts = 1 / dpts
        else:
            import ipdb; ipdb.set_trace()
        return dpts
    
    def read_dpt(self, index):
        dpt_path = self.dpt_files[index]
        dpt = np.asarray(np.load(dpt_path)).astype(np.float32)
        if np.isnan(dpt).any(): dpt[np.isnan(dpt)] = dpt[np.isnan(dpt)==False].max()
        
        # ################
        # log depth
        # dpt = np.clip(dpt, 0.5, 80.)
        # dpt = (np.log(dpt) - np.log(0.5))/(np.log(80.) - np.log(0.5))
        # ################
        
        # ################
        # official depth
        # if self.cfg.get('depth_norm', 'simp') == 'simp':
        #     dpt_min, dpt_max = np.percentile(dpt, 2.), np.percentile(dpt, 98.)
        #     if dpt_max - dpt_min < 1e-6: dpt_max = dpt_min + 2e-6
        #     dpt = np.clip(dpt, dpt_min, dpt_max)
        #     dpt = (dpt - dpt_min) / (dpt_max - dpt_min)
        # elif self.cfg.get('depth_norm', 'simp') == 'log':
        #     dpt = np.clip(dpt, 0.5, 80.)
        #     dpt = (np.log(dpt) - np.log(0.5))/(np.log(80.) - np.log(0.5))
        # elif self.cfg.get('depth_norm', 'simp') == 'disp':
        #     dpt = np.clip(dpt, 1., None)
        #     dpt = 1 / dpt
        # else:
        #     import ipdb; ipdb.set_trace()
        # dpt = np.clip(dpt, 0.1, None)
        # dpt_min, dpt_max = np.percentile(dpt, 2.), np.percentile(dpt, 98.)
        # if dpt_max - dpt_min <= 1e-6: dpt_max = dpt_min + 2e-6
        # dpt = np.clip(dpt, dpt_min, dpt_max)
        # dpt = (dpt - dpt_min) / (dpt_max - dpt_min)
        # ################
        return dpt.astype(np.float32)
    
    def read_rgb_name(self, index: int) -> str:
        frame_idx, seq_ids = self.metas[index]
        rgb_path = self.rgb_files[frame_idx]
        return rgb_path.split('/')[-4] + '_' + rgb_path.split('/')[-3] + '_' + rgb_path.split('/')[-2] + '_' +  os.path.basename(rgb_path)
    
from lib.dataset.depth_estimation.depth_estimation import Dataset as BaseDataset
from lib.dataset.depth_estimation.depth_estimation import *
from os.path import join
import os
from torchvision.transforms import Compose
import pandas as pd
from tqdm import tqdm
import glob

class Dataset(BaseDataset):
    def build_transforms(self):
        self.transform = Compose([
            # Resize(
            #     width=None,
            #     height=None,
            #     resize_target=True if self.cfg.split == 'train' else False,
            #     keep_aspect_ratio=True,
            #     ensure_multiple_of=14,
            #     resize_method='lower_bound',
            #     image_interpolation_method=cv2.INTER_AREA,
            # ),
            # NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
    
    def build_metas(self):
        self.dataset_name = 'arkitscenes'
        # data_root = self.cfg.data_root
        # meta_files = [line.strip() for line in open(join(data_root, self.cfg.split_path), 'r').readlines()]
        # self.rgb_files = []
        # self.depth_files = []
        # for meta_file in meta_files:
        #     rgb_file, depth_file, _ = meta_file.split(' ')
        #     self.rgb_files.append(join(data_root, rgb_file))
        #     self.depth_files.append(join(data_root, depth_file))
        meta_file = pd.read_csv(self.cfg.meta_file)
        self.scene_direction = {str(meta_file['video_id'][idx]): meta_file['sky_direction'][idx] for idx in range(len(meta_file['video_id']))}
        split = 'Training' if self.cfg.split == 'train' else 'Validation'
        scenes = sorted(os.listdir(self.cfg.data_root + '/' + split))
        if 'DEBUG' in os.environ and os.environ['DEBUG'] == 'HTCODE': scenes = scenes[:20]
        data_root = self.cfg.data_root + '/' + split
        self.rgb_files, self.depth_files, self.low_files, self.conf_files = [], [], [], []
        split_file_lines = open(self.cfg.split_file, 'r').readlines()
        for line in split_file_lines:
            rgb_file, depth_file, low_file = line.strip().split(' ')
            self.rgb_files.append(join(data_root, rgb_file))
            self.depth_files.append(join(data_root, depth_file))
            self.low_files.append(join(data_root, low_file))
            self.conf_files.append(join(data_root, low_file.replace('lowres_depth', 'confidence')))
        # self.rgb_files, self.depth_files, self.low_files = [], [], []
        # for scene in tqdm(scenes):
        #     if scene not in self.scene_direction:
        #         continue
        #     if self.cfg.rotate == 'lr' and (self.scene_direction[scene] == 'Up' or self.scene_direction[scene] == 'Down'):  
        #         continue
        #     if self.cfg.rotate == 'ud' and (self.scene_direction[scene] == 'Left' or self.scene_direction[scene] == 'Right'):
        #         continue
        #     rgb_files = sorted(glob.glob(os.path.join(data_root, scene, 'wide', '*.png')))
        #     dpt_files = sorted(glob.glob(os.path.join(data_root, scene, 'highres_depth', '*.png')))
        #     low_files = sorted(glob.glob(os.path.join(data_root, scene, 'lowres_depth', '*.png')))
        #     assert(len(rgb_files) == len(dpt_files))
        #     assert(len(rgb_files) == len(low_files))
        #     self.rgb_files += rgb_files
        #     self.depth_files += dpt_files
        #     self.low_files += low_files
        frames = self.cfg.get('frames', [0, -1, 1])
        s, e, i = frames[0], frames[1], frames[2]
        e = min(e, len(self.rgb_files)) if e != -1 else len(self.rgb_files)
        self.rgb_files = self.rgb_files[s:e:i]
        self.depth_files = self.depth_files[s:e:i]
        self.low_files = self.low_files[s:e:i]
        
    def __getitem__(self, index):
        ret_dict = super().__getitem__(index)
        
        scene = self.rgb_files[index].split('/')[-3]
        direction = self.scene_direction[scene]
        
        # lowres_depth = np.asarray(imageio.imread(self.low_files[index]) / 1000.).astype(np.float32)
        lowres_depth = ret_dict['lowres_depth'][0]
        lowres_depth = Dataset.rotate_image(lowres_depth, direction)
        
        confidence = ret_dict['confidence'][0]
        confidence = Dataset.rotate_image(confidence, direction)
        
        image = ret_dict['image'].transpose(1, 2, 0)
        image = Dataset.rotate_image(image, direction)
        
        depth = ret_dict['depth'][0]
        depth = Dataset.rotate_image(depth, direction)
        
        mask = ret_dict['mask'][0]
        mask = Dataset.rotate_image(mask, direction)
        
        ret_dict.update({
            'image': image.transpose(2, 0, 1),
            'depth': depth[None],
            'lowres_depth': lowres_depth[None],
            'mask': mask[None],
            'confidence': confidence[None]
        })
        return ret_dict
    
    @staticmethod
    def rotate_image(img, direction):
        if direction == 'Up':
            pass
        elif direction == 'Left':
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif direction == 'Right':
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif direction == 'Down':
            img = cv2.rotate(img, cv2.ROTATE_180)
        else:
            raise Exception(f'No such direction (={direction}) rotation')
        return img
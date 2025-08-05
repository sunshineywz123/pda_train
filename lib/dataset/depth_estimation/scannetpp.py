from tqdm.auto import tqdm
import cv2
from lib.utils.pylogger import Log
from trdparties.colmap.read_write_model import read_images_text
from lib.dataset.depth_estimation.depth_estimation import *
from os.path import join
from torchvision.transforms import Compose
from tqdm import tqdm
from lib.dataset.depth_estimation.depth_estimation import Dataset as BaseDataset

class Dataset(BaseDataset):
    def build_metas(self):
        self.dataset_name = 'scannetpp_{self.cfg.scene}'
        root_dir = join(self.cfg.data_root, self.cfg.scene, 'iphone')
        images = read_images_text(join(root_dir, 'colmap/images.txt'))
        img_names = [] 
        for id in images:
            img_names.append(images[id].name)
        img_names = sorted(img_names)
        self.rgb_files, self.depth_files, self.low_files = [], [], []
        for img_name in tqdm(img_names):
            self.rgb_files.append(join(root_dir, 'rgb', img_name))
            self.depth_files.append(join(root_dir, 'render_depth', img_name[:-4] + '.png'))
            self.low_files.append(join(root_dir, 'depth', img_name[:-4] + '.png'))
        
        frame_len = len(self.rgb_files)
        frames = self.cfg.get('frames', [0, -1, 1])
        start, end, step = frames
        end = frame_len if end == -1 else end
        
        self.rgb_files = self.rgb_files[start:end:step]
        self.depth_files = self.depth_files[start:end:step]
        self.low_files = self.low_files[start:end:step]
        Log.info(f'Scannetpp {self.cfg.scene}: {len(self.rgb_files)} frames in total.')
        
    def build_transforms(self):
        if self.cfg.split == 'train':
            train_crop_size = self.cfg.get('train_crop_size', 630)
            down_scale = self.cfg.get('down_scale', 7.5)
            width = self.cfg.get('width', None)
            height = self.cfg.get('height', None)
            resize_target = self.cfg.get('resize_target', True)
            Log.info(f"Using train width {width}")
            Log.info(f"Using train height {height}")
            Log.info(f"Using crop size {train_crop_size}")
            Log.info(f"Using down scale {down_scale}")
            Log.info(f"Resize target: {resize_target}")
            self.transform = Compose([
                                    Resize(width=width,
                                             height=height,
                                             resize_target=resize_target,
                                             keep_aspect_ratio=True,
                                             ensure_multiple_of=1,
                                             resize_method='lower_bound',
                                             image_interpolation_method=cv2.INTER_AREA),
                                    PrepareForNet(), 
                                    Crop(train_crop_size, down_scale=down_scale)
                                      ])
        else:
            ensure_multiple_of = self.cfg.get('ensure_multiple_of', 14)
            width = self.cfg.get('width', None)
            height = self.cfg.get('height', None)
            resize_target = self.cfg.get('resize_target', False)
            Log.info(f"Using ensure_multiple_of {ensure_multiple_of}")
            Log.info(f"Using test width {width}")
            Log.info(f"Using test height {height}")
            Log.info(f"Resize target: {resize_target}")
            self.transform = Compose([
                Resize(
                    width=width,
                    height=height,
                    resize_target=resize_target,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=ensure_multiple_of,
                    resize_method='lower_bound',
                    image_interpolation_method=cv2.INTER_AREA,
                ),
                PrepareForNet()])
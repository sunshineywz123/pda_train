from tqdm.auto import tqdm
import cv2
from lib.utils.pylogger import Log
from trdparties.colmap.read_write_model import read_images_text, read_images_binary
from lib.dataset.depth_estimation.depth_estimation import *
from os.path import join
from torchvision.transforms import Compose
from tqdm import tqdm
from lib.dataset.depth_estimation.depth_estimation import Dataset as BaseDataset

class Dataset(BaseDataset):
    def build_metas(self):
        self.dataset_name = 'scannetpp_{self.cfg.scene}'
        root_dir = join(self.cfg.data_root, self.cfg.scene, 'merge')
        low_dir = join(self.cfg.data_root, self.cfg.scene, 'iphone/depth')
        mesh_depth_dir = join(self.cfg.data_root, self.cfg.scene, 'iphone/render_depth')
        version = 'v1'
        images = read_images_binary(join(root_dir, f'depth/{version}/colmap/images.bin'))
        img_names = [] 
        for id in images: 
            if 'iphone' in images[id].name:
                img_names.append(images[id].name)
        img_names = sorted(img_names)
        self.rgb_files, self.depth_files, self.low_files = [], [], []
        for img_name in tqdm(img_names):
            self.rgb_files.append(join(root_dir, 'images', img_name))
            self.depth_files.append(join(root_dir, f'depth/{version}/npz', img_name[7:-4] + '.npz'))
            # self.depth_files.append(join(mesh_depth_dir, img_name[7:-5] + '0.png'))
            self.low_files.append(join(low_dir, img_name[7:-4] + '.png'))
        
        frame_len = len(self.rgb_files)
        frames = self.cfg.get('frames', [0, -1, 1])
        frames = [0, -1, 1]
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
            train_crop_size = self.cfg.get('train_crop_size', 630)
            down_scale = self.cfg.get('down_scale', 7.5)
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
                PrepareForNet(),
                # Crop(train_crop_size, down_scale=down_scale, center=True)
                ])
            
    def read_depth(self, index, depth=None):
        start_time = time.time()
        if self.depth_files[index].endswith('.npz'): depth = np.load(self.depth_files[index])['data']
        else: depth = np.asarray(imageio.imread(self.depth_files[index]) / 1000.).astype(np.float32)
        
        end_time = time.time()
        if end_time - start_time > 1: Log.info(f'Long time to read {self.depth_files[index]}: {end_time - start_time}')
        min_val = np.percentile(depth, 2.)
        max_val = np.percentile(depth, 98.)
        valid_mask = np.logical_and(depth > min_val, depth < max_val)
        return depth, valid_mask.astype(np.uint8)
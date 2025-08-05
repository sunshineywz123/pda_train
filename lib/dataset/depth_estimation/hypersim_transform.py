from lib.dataset.depth_estimation.depth_estimation import *
from torchvision.transforms import Compose
from lib.dataset.depth_estimation.hypersim import Dataset as BaseDataset

class Dataset(BaseDataset):
    def build_transforms(self):
        if 'train' in self.cfg.split:
            train_crop_size = self.cfg.get('train_crop_size', 630)
            down_scale = self.cfg.get('down_scale', 7.5)
            width = self.cfg.get('width', None)
            height = self.cfg.get('height', None)
            resize_target = self.cfg.get('resize_target', True)
            ensure_multiple_of = self.cfg.get('ensure_multiple_of', 1)
            Log.info(f"Using train width {width}")
            Log.info(f"Using train height {height}")
            Log.info(f"Using crop size {train_crop_size}")
            Log.info(f"Using down scale {down_scale}")
            Log.info(f"Using train ensure_multiple_of {ensure_multiple_of}")
            Log.info(f"Resize target: {resize_target}")
            transforms = [
                Resize(width=width,
                       height=height,
                       resize_target=resize_target,
                       keep_aspect_ratio=True,
                       ensure_multiple_of=ensure_multiple_of,
                       resize_method='lower_bound',
                       image_interpolation_method=cv2.INTER_AREA),
                PrepareForNet(), 
            ]
            if train_crop_size != -1: transforms.append(Crop(train_crop_size, down_scale=down_scale))
            self.transform = Compose(transforms)
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
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=ensure_multiple_of,
                    resize_method='lower_bound',
                    image_interpolation_method=cv2.INTER_AREA,
                ),
                PrepareForNet()])
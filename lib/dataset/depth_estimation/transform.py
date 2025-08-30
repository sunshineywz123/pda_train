import cv2
import numpy as np
from lib.utils.pylogger import Log
import torch
import torch.nn.functional as F

class Rotate(object):
    """Prepare sample for usage as network input.
    """

    def __init__(self):
        pass

    def __str__(self):
        return "Rotate"

    def __repr__(self):
        return "Rotate"

    def __call__(self, sample):
        assert 'direction' in sample, 'No direction in sample'
        if sample['direction'] == 'Up':
            pass
        else:
            if sample['direction'] == 'Left':
                rotate_option = cv2.ROTATE_90_CLOCKWISE
            elif sample['direction'] == 'Right':
                rotate_option = cv2.ROTATE_90_COUNTERCLOCKWISE
            elif sample['direction'] == 'Down':
                rotate_option = cv2.ROTATE_180
            else:
                raise Exception(f'No such direction (={sample["direction"]}) rotation')
            sample['image'] = cv2.rotate(sample['image'], rotate_option)
            if 'mask' in sample:
                sample['mask'] = cv2.rotate(sample['mask'], rotate_option)
            if 'depth' in sample:
                sample['depth'] = cv2.rotate(sample['depth'], rotate_option)
            if 'lowres_depth' in sample:
                sample['lowres_depth'] = cv2.rotate(sample['lowres_depth'], rotate_option)
            if 'confidence' in sample:
                sample['confidence'] = cv2.rotate(sample['confidence'], rotate_option)
        return sample
    
class UnPrepareForNet(object):
    def __init__(self):
        pass
    def __str__(self):
        return "UnPrepareForNet"
    def __repr__(self):
        return "UnPrepareForNet"
    def __call__(self, sample):
        image = np.transpose(sample["image"], (1, 2, 0))
        sample['image'] = image
        if "mask" in sample:
            sample["mask"] = sample["mask"][0]
        if "confidence" in sample:
            sample["confidence"] = sample["confidence"][0]
        if "depth" in sample:
            sample["depth"] = sample["depth"][0]
        if "lowres_depth" in sample:
            sample["lowres_depth"] = sample["lowres_depth"][0]
        if "semseg_mask" in sample:
            sample["semseg_mask"] = sample["semseg_mask"][0]
        if "semantic" in sample:
            sample["semantic"] = sample["semantic"][0]
        if "mesh_depth" in sample:
            sample["mesh_depth"] = sample["mesh_depth"][0]
        return sample
class PrepareForNet(object):
    """Prepare sample for usage as network input.
    """

    def __init__(self):
        pass

    def __str__(self):
        return "PrepareForNet"

    def __repr__(self):
        return "PrepareForNet"

    def __call__(self, sample):
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)

        if "mask" in sample:
            sample["mask"] = sample["mask"].astype(np.uint8)
            sample["mask"] = np.ascontiguousarray(sample["mask"])[None]

        if "confidence" in sample:
            sample["confidence"] = sample["confidence"].astype(np.uint8)
            sample["confidence"] = np.ascontiguousarray(
                sample["confidence"])[None]

        if "depth" in sample:
            depth = sample["depth"].astype(np.float32)
            sample["depth"] = np.ascontiguousarray(depth)[None]

        if "mesh_depth" in sample:
            mesh_depth = sample["mesh_depth"].astype(np.float32)
            sample["mesh_depth"] = np.ascontiguousarray(mesh_depth)[None]

        if "semseg_mask" in sample:
            sample["semseg_mask"] = sample["semseg_mask"].astype(np.float32)
            sample["semseg_mask"] = np.ascontiguousarray(
                sample["semseg_mask"])[None]
            
        if "semantic" in sample:
            sample["semantic"] = sample["semantic"].astype(np.uint8)
            sample["semantic"] = np.ascontiguousarray(sample["semantic"])[None]

        if "lowres_depth" in sample:
            lowres_depth = sample["lowres_depth"].astype(np.float32)
            sample["lowres_depth"] = np.ascontiguousarray(lowres_depth)[None]

        return sample

def cv2_resize(image, size, interpolation=cv2.INTER_LINEAR):
    return cv2.resize(image, size, interpolation=interpolation)[None]

import os
from lib.utils.depth_utils import interp_depth_rgb, GenerateSpotMask, bilateralFilter
def random_simu(depth, tar_size, use_bi=True):
    img_w, img_h = tar_size
    depth = cv2.resize(depth, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    if np.random.random() < 0.2:
        return depth.astype(np.float32)
    if img_w == img_h: dist_coef=0.
    else: dist_coef=2e-5
    orig_depth = depth
    rand = np.random.randn(img_h * img_w).reshape(img_h, img_w)
    rand = rand * 0.005 - 0.01
    depth_min = np.min(depth)
    depth = depth + rand * depth
    if np.random.random() < float(os.environ.get('spot_mask_prob', 0.5)):
        return depth.astype(np.float32)
    spot_mask = GenerateSpotMask(img_h, img_w, stride=3, dist_coef=dist_coef)
    sparse_depth = np.zeros_like(depth)
    spot_mask = spot_mask==1.
    sparse_depth[spot_mask] = depth[spot_mask]
    
    sparse_depth = interp_depth_rgb(
        sparse_depth,
        orig_depth,
        speed=5,
        k = 4
    )
    if use_bi:
        bi_sparse_depth = bilateralFilter(orig_depth, 5, 0.01, 50., sparse_depth)
        return bi_sparse_depth.astype(np.float32)
    else:
        return sparse_depth.astype(np.float32)
    import matplotlib.pyplot as plt 
    plt.subplot(231)
    plt.imshow(orig_depth)
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(depth)
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(sparse_depth)
    plt.axis('off')
    plt.subplot(234)
    plt.imshow(spot_mask)
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(bi_sparse_depth)
    plt.axis('off')
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig('test.jpg', dpi=300)
    import ipdb; ipdb.set_trace()
    return bi_sparse_depth



class CompLowRes(object):
    def __init__(self, height=None, width=None, ranges=None, range_prob=None, interpolation='cv2'):
        self._height = height
        self._width = width
        self._ranges = ranges
        self._range_prob = range_prob
        self._interpolation = interpolation

    def __str__(self):
        return "SimLowRes: height: {}, width: {}, ranges: {}, range_prob: {}".format(self._height, self._width, self._ranges, self._range_prob)

    def __call__(self, sample):
        try:
            rgb = np.transpose(sample['image'], (1, 2, 0))  # 将图像从CHW格式转换为HWC格式
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)    # 将RGB图像转换为灰度图像
            lowres_depth = sample['lowres_depth'][0]        # 获取低分辨率深度图的第一个通道
            if (lowres_depth != 0).sum() <= 10:             # 如果低分辨率深度图中非零像素点小于等于10
                sample['lowres_depth'] = cv2_resize(sample['depth'][0], (lowres_depth.shape[1], lowres_depth.shape[0]), interpolation=cv2.INTER_LINEAR)  # 用原始深度图resize填充
                return sample                               # 返回处理后的sample
            
            if gray.shape[0] != lowres_depth.shape[0] or gray.shape[1] != lowres_depth.shape[1]:  # 如果灰度图和低分辨率深度图的尺寸不一致
                gray = cv2.resize(gray, (lowres_depth.shape[1], lowres_depth.shape[0]), interpolation=cv2.INTER_LINEAR)  # 将灰度图resize到低分辨率深度图的尺寸
            sparse_depth = interp_depth_rgb(
                lowres_depth,                               # 输入低分辨率深度图
                gray,                                      # 输入灰度图
                speed=5,                                   # 插值速度参数
                k = 4                                      # 最近邻点数参数
            )
            if rgb.shape[0] < sparse_depth.shape[0] or rgb.shape[1] < sparse_depth.shape[1]:  # 如果原始图像尺寸小于插值后的深度图
                sparse_depth = cv2.resize(sparse_depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)  # 将插值后的深度图resize到原始图像尺寸
            # import ipdb; ipdb.set_trace()
            # bi_sparse_depth = bilateralFilter(gray, 5, 0.01, 50., sparse_depth)
            sample['lowres_depth'] = sparse_depth[None]  # 将插值后的稀疏深度图增加一个维度后赋值给sample的'lowres_depth'键 ##knn差值后的深度
            sample['sparse_depth'] = lowres_depth[None]  # 将原始低分辨率深度图增加一个维度后赋值给sample的'sparse_depth'键 ##lidar稀疏深度
        except:
            import ipdb; ipdb.set_trace()
        return sample

class SimLowRes(object):
    def __init__(self, height=None, width=None, ranges=None, range_prob=None, interpolation='cv2', use_bi=True):
        self._height = height
        self._width = width
        self._ranges = ranges
        self._range_prob = range_prob
        self._interpolation = interpolation
        self._use_bi = use_bi

    def __str__(self):
        return "SimLowRes: height: {}, width: {}, ranges: {}, range_prob: {}".format(self._height, self._width, self._ranges, self._range_prob)

    def __call__(self, sample):
        assert 'lowres_depth' not in sample, 'lowres depth in sample'
        if self._height is not None: tar_size = (self._width, self._height)
        else:
            assert sample['depth'].shape[1] == sample['depth'].shape[2]
            choice = np.random.choice(np.arange(len(self._range_prob)), p=self._range_prob)
            if choice == 0: tar_size = (self._ranges[0], self._ranges[0])
            elif choice == 2: tar_size = (self._ranges[1], self._ranges[1])
            elif choice == 1:
                tar_size = int(np.random.random() * (self._ranges[1] - self._ranges[0]) + self._ranges[0])
                tar_size = (tar_size, tar_size)
        if self._interpolation == 'cv2': 
            sample['lowres_depth'] = cv2_resize(sample['depth'][0], tar_size, interpolation=cv2.INTER_LINEAR)[None]
        elif self._interpolation == 'random_simu':
            sample['lowres_depth'] = random_simu(sample['depth'][0], tar_size, use_bi=self._use_bi)[None]
        return sample



class SimuCarLidar(object):
    def __init__(self, tar_lines=64, reserve_ratio=0.5, min_pitch=-0.07):
        self._tar_lines = tar_lines
        self._reserve_ratio = reserve_ratio
        self._min_pitch = min_pitch
        self.ixt = np.array([[640., 0., 315], [0., 640., 210.], [0., 0., 1.]])
        
    def __str__(self):
        return "SimuCarLidar: tar_lines: {}, reserve_ratio: {}, min_pitch: {}".format(self._tar_lines, self._reserve_ratio, self._min_pitch)

    def __call__(self, sample):
        assert 'lowres_depth' not in sample, 'lowres depth in sample'
        # import pdb;pdb.set_trace()
        depth_map = sample['depth'][0]
        v, u = np.nonzero(depth_map)
        z = depth_map[v, u]
        
        points = np.linalg.inv(self.ixt) @ (np.vstack([u, v, np.ones_like(u)]) * z)
        points = points.transpose([1, 0])
        
        scan_y = points[:, 1]
        distance = np.linalg.norm(points, 2, axis=1)
        pitch = np.arcsin(scan_y / distance)
        num_points = np.shape(pitch)[0]
        pitch = np.reshape(pitch, (num_points, 1))
        max_pitch = np.max(pitch)
        min_pitch = np.min(pitch)
        
        input_lines = depth_map.shape[0]
        num_lines = int(input_lines * (self._min_pitch - max_pitch) / (min_pitch - max_pitch)) - 1
        keep_ratio = self._tar_lines / num_lines
        
        angle_interval = (max_pitch - self._min_pitch) / num_lines
        angle_label = np.round((pitch - min_pitch) / angle_interval)
        
        sampling_mask = angle_label % int((1.0 / keep_ratio)) == 0
        sampling_mask = sampling_mask & (pitch >= self._min_pitch)
        
        final_mask=np.zeros_like(depth_map)
        final_mask[v,u]=sampling_mask[:,0]
        final_mask=final_mask.astype(np.bool_)
        # final_mask = sampling_mask.reshape(depth_map.shape)
        
        random_mask = np.random.random(final_mask.shape) < self._reserve_ratio
        
        grid_mask = np.zeros_like(final_mask, dtype=bool)
        grid_mask[:, ::2] = True
        
        
        max_val = 80
        trunc_val = 50
        distance_drop = ((depth_map - trunc_val) / (max_val - trunc_val))**3 + np.random.random(depth_map.shape) 
        final_mask = final_mask & grid_mask & random_mask & (distance_drop < 1.)
        
        output_depth = np.zeros_like(depth_map)
        output_depth[final_mask] = depth_map[final_mask]
        
        sample['lowres_depth'] = output_depth[None]
        return sample
        
        
class SimuCarLidarApollo(object):
    def __init__(self, tar_lines=64, reserve_ratio=0.5, min_pitch=-0.07):
        self._tar_lines = tar_lines
        self._reserve_ratio = reserve_ratio
        self._min_pitch = min_pitch
        self.ixt = np.array([[2015., 0., 810.], [0., 2015., 540.], [0., 0., 1.]])
        self.ixt[:2] *= 0.62222222
        
    def __str__(self):
        return "SimuCarLidar: tar_lines: {}, reserve_ratio: {}, min_pitch: {}".format(self._tar_lines, self._reserve_ratio, self._min_pitch)

    def __call__(self, sample):
        assert 'lowres_depth' not in sample, 'lowres depth in sample'
        depth_map = sample['depth'][0]
        v, u = np.nonzero(depth_map)
        z = depth_map[v, u]
        
        points = np.linalg.inv(self.ixt) @ (np.vstack([u, v, np.ones_like(u)]) * z)
        points = points.transpose([1, 0])
        
        scan_y = points[:, 1]
        distance = np.linalg.norm(points, 2, axis=1)
        pitch = np.arcsin(scan_y / distance)
        num_points = np.shape(pitch)[0]
        pitch = np.reshape(pitch, (num_points, 1))
        max_pitch = np.max(pitch)
        min_pitch = np.min(pitch)
        
        input_lines = depth_map.shape[0]
        num_lines = int(input_lines * (self._min_pitch - max_pitch) / (min_pitch - max_pitch)) - 1
        keep_ratio = self._tar_lines / num_lines
        
        angle_interval = (max_pitch - self._min_pitch) / num_lines
        angle_label = np.round((pitch - min_pitch) / angle_interval)
        
        sampling_mask = angle_label % int((1.0 / keep_ratio)) == 0
        sampling_mask = sampling_mask & (pitch >= self._min_pitch)
        
        final_mask = sampling_mask.reshape(depth_map.shape)
        
        random_mask = np.random.random(final_mask.shape) < self._reserve_ratio
        
        grid_mask = np.zeros_like(final_mask, dtype=bool)
        grid_mask[:, ::3] = True
        
        
        max_val = 80
        trunc_val = 50
        distance_drop = ((depth_map - trunc_val) / (max_val - trunc_val))**3 + np.random.random(depth_map.shape) 
        final_mask = final_mask & grid_mask & random_mask & (distance_drop < 1.)
        
        output_depth = np.zeros_like(depth_map)
        output_depth[final_mask] = depth_map[final_mask]
        
        # import imageio
        # imageio.imwrite('test.png', (final_mask * 255).astype(np.uint8))
        # imageio.imwrite('test.jpg', (sample['image'].transpose(1, 2, 0) * 255.).astype(np.uint8))
        # import ipdb; ipdb.set_trace()
        sample['lowres_depth'] = output_depth[None]
        return sample


class Crop(object):
    """Crop sample for batch-wise training. Image is of shape CxHxW
    """

    def __init__(self, size, down_scale=7.5, center=False, down_scales=None, down_scale_prob=None):
        self.size = size
        self._center = center
        # downs_scale = 1, 2, 4, 7.5, 8
        assert down_scale in [-1, 1, 2, 3.75, 4, 7.5, 8, 14, 28], 'Wrong down_scale'
        self._down_scale = down_scale
        local_dict = {-1: 1, 1: 1, 2: 2, 3.75:15, 4: 4, 8: 8, 7.5: 15, 14: 1, 28: 1}
        self._local_dict = local_dict
        self._ensure_round = local_dict[down_scale]
        self._down_scales = down_scales
        self._down_scale_prob = down_scale_prob
        # Log.info(f"Crop size: {size}, down_scale: {down_scale}, center: {center}")

    def __str__(self):
        return "RandomCrop: size: {}, down_scale: {}, center: {}".format(self.size, self._down_scale, self._center)


    def get_bbox(self, sample, h, w, ensure_round):
        if self.size == -1: return 0, 0, h, w
        assert h >= self.size and w >= self.size, 'Wrong size'
        h_start = np.random.randint(0, h - self.size)
        w_start = np.random.randint(0, w - self.size)
        if 'lowres_depth' in sample:
            h_start = h_start // ensure_round * ensure_round
            w_start = w_start // ensure_round * ensure_round
        h_end = h_start + self.size
        w_end = w_start + self.size
        return h_start, w_start, h_end, w_end

    def __call__(self, sample):
        if self._down_scales is not None:
            down_scale = np.random.choice(self._down_scales, p=self._down_scale_prob)
            ensure_round = self._local_dict[down_scale]
        else:
            down_scale = self._down_scale
            ensure_round = self._ensure_round
        if isinstance(self.size, int):
            h_start, w_start, h_end, w_end = self.get_bbox(sample, sample['image'].shape[-2], sample['image'].shape[-1], ensure_round)
        else:
            h_start, w_start, h_end, w_end = self.size
        # if self.size == -1:
        #     return sample
        # h, w = sample['image'].shape[-2:]
        # assert h >= self.size and w >= self.size, 'Wrong size'
        # h_start, w_start = np.random.randint(
        #     0, h - self.size), np.random.randint(0, w - self.size)
        # if 'lowres_depth' in sample:
        #     h_start = h_start // self._ensure_round * self._ensure_round
        #     w_start = w_start // self._ensure_round * self._ensure_round
        # h_end = h_start + self.size
        # w_end = w_start + self.size
        # if self._center:
        #     h_start = (h - self.size) // 2
        #     w_start = (w - self.size) // 2
        #     h_end = h_start + self.size
        #     w_end = w_start + self.size


        sample['image'] = sample['image'][:, h_start: h_end, w_start: w_end]

        if "depth" in sample:
            sample["depth"] = sample["depth"][:,
                                              h_start: h_end, w_start: w_end]

        if "mask" in sample:
            sample["mask"] = sample["mask"][:, h_start: h_end, w_start: w_end]

        if "semseg_mask" in sample:
            sample["semseg_mask"] = sample["semseg_mask"][:,
                                                          h_start: h_end, w_start: w_end]

        if 'lowres_depth' in sample:
            ds = down_scale
            sample['lowres_depth'] = sample['lowres_depth'][:, int(
                h_start/ds): int(h_end/ds), int(w_start/ds): int(w_end/ds)]

        if 'confidence' in sample:
            ds = down_scale
            sample['confidence'] = sample['confidence'][:, int(
                h_start/ds): int(h_end/ds), int(w_start/ds): int(w_end/ds)]
    
        return sample


class Resize(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width=None,
        height=None,
        resize_ratio=None,
        resize_target=False,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.width = width
        self.height = height
        self.__width = width
        self.__height = height
        self.__resize_ratio = resize_ratio
        assert( 
            (width is not None and height is not None) or resize_ratio is not None
        )
        assert(
            (width is None and height is None) or resize_ratio is None
        )

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method
        # Log.info(f"Resize width: {width}, height: {height}, resize_target: {resize_target}, keep_aspect_ratio: {keep_aspect_ratio}, ensure_multiple_of: {ensure_multiple_of}, resize_method: {resize_method}")

    def __str__(self):
        return "Resize: width: {}, height: {}, resize_target: {}, keep_aspect_ratio: {}, ensure_multiple_of: {}, resize_method: {}".format(self.__width, self.__height, self.__resize_target, self.__keep_aspect_ratio, self.__multiple_of, self.__resize_method)

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)
        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of)
                 * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of)
                 * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        if self.__resize_ratio is not None:
            __height = int(height * self.__resize_ratio)
            __width = int(width * self.__resize_ratio)
        else:
            __height = self.__height if self.__height is not None else height
            __width = self.__width if self.__width is not None else width
        scale_height = __height / height
        scale_width = __width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(
                f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(
            sample["image"].shape[1], sample["image"].shape[0]
        )
        if width == sample['image'].shape[1] and height == sample['image'].shape[0]:
            return sample
        Log.debug(
            'Resize: {} -> {}'.format(sample["image"].shape, (height, width)))
        # resize sample
        sample["image"] = cv2.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method,
        )

        # if 'lowres_depth' in sample:
        #     width_low, height_low = int(width / 7.5), int(height / 7.5)
        #     sample['lowres_depth'] = cv2.resize(
        #         sample['lowres_depth'],
        #         (width_low, height_low),
        #         interpolation=cv2.INTER_LINEAR
        #     )

        if self.__resize_target:
            Log.debug('Resize target')
            if "disparity" in sample:
                sample["disparity"] = cv2.resize(
                    sample["disparity"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            if "depth" in sample:
                sample["depth"] = cv2.resize(
                    sample["depth"], (width,
                                      height), interpolation=cv2.INTER_NEAREST
                )

            if "mesh_depth" in sample:
                sample["mesh_depth"] = cv2.resize(
                    sample["mesh_depth"], (width,
                                           height), interpolation=cv2.INTER_NEAREST
                )

            if "semseg_mask" in sample:
                sample["semseg_mask"] = F.interpolate(torch.from_numpy(sample["semseg_mask"]).float()[
                                                      None, None, ...], (height, width), mode='nearest').numpy()[0, 0]

            if "semantic" in sample:
                sample["semantic"] = cv2.resize(
                    sample["semantic"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                

            if "mask" in sample:
                sample["mask"] = cv2.resize(
                    sample["mask"].astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                # sample["mask"] = sample["mask"].astype(bool)

        # print(sample['image'].shape, sample['depth'].shape)
        return sample

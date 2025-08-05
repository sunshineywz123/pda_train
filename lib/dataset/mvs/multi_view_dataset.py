# Default dataset for loading volumetric videos
# Expected formats: easymocap camera parameters & images folder
# multiview dataset format:
# intri.yml
# extri.yml
# images/
#     00/
#         000000.jpg
#         000001.jpg
#         ...
#     01/
#         000000.jpg
#         000001.jpg
#         ...

# monocular dataset format:
# cameras/
#     00/
#         intri.yml
#         extri.yml
# images/
#     00/
#         000000.jpg
#         000001.jpg
#         ...
# will prepare all camera parameters before the actual data loading
# will perform random samplinng on the rays (either patch or rays)

# The exposed apis should be as little as possible
import os
import cv2  # for undistortion
import torch
import random
import numpy as np
from glob import glob
from typing import List
from functools import lru_cache, partial
from torch.utils.data import Dataset, get_worker_info

from lib.utils.epipolar_utils import compute_masks_pluckers
from lib.utils.depth_utils import Normalization, normalize_depth
from lib.utils.image_utils import get_xywh_from_hwc, crop_using_xywh

from easyvolcap.engine import DATASETS, cfg, args
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.timer_utils import timer
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.easy_utils import read_camera
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.math_utils import affine_inverse, affine_padding
from easyvolcap.utils.bound_utils import get_bound_2d_bound, monotonic_near_far
from easyvolcap.utils.cam_utils import average_c2ws, Sourcing, compute_camera_similarity, compute_camera_zigzag_similarity
from easyvolcap.utils.data_utils import DataSplit, UnstructuredTensors, load_resize_undist_ims_bytes, load_image_from_bytes, as_torch_func, to_cuda, to_cpu, to_tensor, export_pts, load_pts, decode_crop_fill_ims_bytes, decode_fill_ims_bytes, load_resize_undist_im_bytes, save_image


class MultiViewDataset(Dataset):
    def __init__(self,
                 # Dataset intrinsic properties
                 data_root: str,  # this must be configured
                 split: str = DataSplit.TRAIN.name,  # dynamically generated

                 # The frame number & image size should be inferred from the dataset
                 ratio: float = 1.0,  # use original image size
                 center_crop_size: List[int] = [-1, -1],  # center crop image to this size, after resize
                 view_sample: List = [0, None, 1],  # begin, end, step
                 frame_sample: List = [0, None, 1],  # begin, end, step

                 # Other default configurations
                 intri_file: str = 'intri.yml',
                 extri_file: str = 'extri.yml',
                 images_dir: str = 'images',
                 cameras_dir: str = 'cameras',  # only when the camera is moving through time
                 ims_pattern: str = '{frame:06d}.jpg',
                 imsize_overwrite: List[int] = [-1, -1],  # overwrite the image size

                 # Camera alignment
                 use_aligned_cameras: bool = False,
                 avg_using_all: bool = False,  # ok enough for now
                 avg_max_count: int = 100,  # prevent slow center of attention computation

                 # Mask related configs
                 masks_dir: str = 'masks',
                 use_masks: bool = False,
                 imbound_crop: bool = False,
                 immask_crop: bool = False,
                 immask_fill: bool = False,

                 # Depth related configs
                 depths_dir: str = 'depths',
                 use_depths: bool = True,  # since we're doing depth estimation, so deafult True

                 # Normal related configs
                 normals_dir: str = 'normals',
                 use_normals: bool = False,

                 # Image preprocessing & formatting
                 dist_opt_K: bool = True,  # use optimized K for undistortion (will crop out black edges), mostly useful for large number of images
                 encode_ext: str = '.jpg',
                 cache_raw: bool = False,

                 # Spacetime based config
                 use_loaded_time: bool = False,  # use the time provided by the datasets, rare
                 duration: float = None,
                 bounds: List[List[float]] = [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
                 near: float = 0.01,  # NOTE: here, near is used as the minimum depth or disparity for normalization
                 far: float = 100,  # NOTE: here, far is used as the maximum depth or disparity for normalization
                 depth_norm: str = Normalization.SIMP.name,  # [minmax, simp, log, disp, nonorm]
                 scale: float = 1.0,  # scale the depth and corresponding camera extrinsics

                 # Dataloading cache config
                 image_cache_maxsize: int = 0,  # 1000 this could be too much
                 preload_images: bool = False,
                 dist_mask: List[bool] = [1] * 5,
                 skip_loading_images: bool = args.type == 'gui',  # for debugging and visualization

                 # Image-based related configs
                 barebone: bool = True,  # this is always True here
                 supply_decoded: bool = True,  # this is always True here
                 n_srcs_list: List[int] = [3],
                 n_srcs_prob: List[int] = [1.0],
                 append_gt_prob: float = 0.0,
                 source_type: str = Sourcing.DISTANCE.name,  # Sourcing.DISTANCE or Sourcing.ZIGZAG
                 closest_using_t: bool = False,  # for backwards compatibility
                 force_sparse_view: bool = False,
                 extra_src_pool: int = 0,  # TODO: Default no extra source pool for now, maybe change this in the future
                 src_view_sample: List[int] = [0, None, 1],
                 latent_factor: int = 8,  # cropping of the image for painless up convolution and skip connections
                 network_factor: int = 8,  # network downsample the image for faster training

                 # Epipolar related configs
                 use_epipolar: bool = False,

                 # Source indexing related configs
                 use_cached_srcs: bool = False,
                 source_dir: str = 'source',
                 src_inds_file: str = 'src_inds.npy',
                 src_sims_file: str = 'src_sims.npy',

                 # Image-based sampler related configs
                 sampler_view_sample: List[int] = [0, None, 1],
                 sampler_frame_sample: List[int] = [0, None, 1],

                 # To remove errors
                 meta_roots: List[str] = None,
                 meta_infos: List[str] = None,
                 ):

        # Global dataset config entries
        self.data_root = data_root
        self.intri_file = intri_file
        self.extri_file = extri_file

        # Camera and alignment configs
        self.avg_using_all = avg_using_all
        self.avg_max_count = avg_max_count
        self.use_aligned_cameras = use_aligned_cameras

        # Data and priors directories
        self.cameras_dir = cameras_dir
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.depths_dir = depths_dir
        self.normals_dir = normals_dir
        self.ims_pattern = ims_pattern

        # Camera and image selection
        self.frame_sample = frame_sample
        self.view_sample = view_sample
        if self.view_sample[1] is not None: self.n_view_total = self.view_sample[1]
        else: self.n_view_total = len(os.listdir(join(self.data_root, self.images_dir)))  # total number of cameras before filtering
        if self.frame_sample[1] is not None: self.n_frames_total = self.frame_sample[1]
        else: self.n_frames_total = min([len(glob(join(self.data_root, self.images_dir, cam, '*'))) for cam in os.listdir(join(self.data_root, self.images_dir))])  # total number of images before filtering

        # Rendering and space carving bounds
        assert not (duration is None and use_loaded_time), "When using loaded time, expect the user to provide the dataset duration"
        self.use_loaded_time = use_loaded_time  # when using loaded time, will normalize to 0-1 with duration
        self.duration = duration if duration is not None else 1.0
        self.bounds = torch.as_tensor(bounds, dtype=torch.float)

        # Depth and disparity normalization
        self.near = near
        self.far = far
        self.depth_norm = Normalization[depth_norm]
        self.scale = scale

        # Compute needed visual hulls & align all cameras loaded
        self.load_cameras()  # load and normalize all cameras (center lookat, align y axis)
        self.select_cameras()  # select repective cameras to use

        # Load the actual data (as encoded jpeg bytes)
        self.split = DataSplit[split]
        self.ratio = ratio  # could be a float (shared ratio) or a list of floats (should match images)
        self.encode_ext = encode_ext
        self.cache_raw = cache_raw  # use raw pixels to further accelerate training
        self.use_depths = use_depths  # use visual hulls as a prior
        self.use_masks = use_masks  # always load mask if using vhulls
        self.use_normals = use_normals  # use normals as a prior
        self.imsize_overwrite = imsize_overwrite  # overwrite loaded image sizes (for enerf)
        self.immask_crop = immask_crop  # maybe crop stored jpeg bytes
        self.immask_fill = immask_fill  # maybe fill stored jpeg bytes
        self.center_crop_size = center_crop_size  # center crop size

        # Distorsion related
        self.dist_mask = dist_mask  # ignore some of the camera parameters
        self.dist_opt_K = dist_opt_K

        self.load_paths()  # load image files into self.ims
        self.skip_loading_images = skip_loading_images
        if not self.skip_loading_images:
            self.load_bytes()  # load image bytes (also load vhulls)
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/

        # Other entries not needed for initialization
        self.imbound_crop = imbound_crop

        # Cache related config
        self.image_cache_maxsize = image_cache_maxsize  # make this smaller to avoid oom
        if image_cache_maxsize:
            self.get_image = lru_cache(image_cache_maxsize)(self.get_image)
            if preload_images:
                pbar = tqdm(total=len(self), desc=f'Preloading raw imgs for {blue(self.data_root)} {magenta(self.split.name)}')
                for v in range(self.n_views):
                    for l in range(self.n_latents):
                        self.get_image(v, l)
                        pbar.update()

        # Epipolar related configs
        self.use_epipolar = use_epipolar

        # Image-based related config
        self.closest_using_t = closest_using_t
        self.src_view_sample = src_view_sample
        assert not self.closest_using_t or self.frame_sample == [0, None, 1] or force_sparse_view, "Should use default frame_sample [0, None, 1] for ibr dataset with `closest_using_t`. Control sampling through sampler.frame_sample and src_view_sample"
        assert self.view_sample == [0, None, 1] or force_sparse_view, "Should use default view_sample [0, None, 1] for ibr dataset. Control sampling through sampler.view_sample and src_view_sample"
        assert not (self.cache_raw and not supply_decoded), "Will always supply decoded source images when cache_raw is enabled for faster sampling, set cache_raw to False to supply jpeg streams"

        # Source related config
        self.use_cached_srcs = use_cached_srcs
        self.source_dir = source_dir
        self.src_inds_file = src_inds_file
        self.src_sims_file = src_sims_file
        self.source_type = Sourcing[source_type]
        # Views are selected and loaded
        # Frames are selected and loaded
        self.load_source_params()
        # Need to build all possible view selections (distance of c2w)
        # - Dot product of v_front - euclidian distance of center
        self.load_source_indices()

        self.n_srcs_list = n_srcs_list if len(n_srcs_list) != 1 or n_srcs_list[0] != 0 else [self.n_views]
        self.n_srcs_prob = n_srcs_prob
        self.extra_src_pool = extra_src_pool
        self.append_gt_prob = append_gt_prob

        # src_inps will come in as decoded bytes instead of jpegs
        self.supply_decoded = supply_decoded
        self.barebone = barebone
        self.latent_factor = latent_factor
        self.network_factor = network_factor

        # Image-based sampler related configs
        self.sampler_view_sample = sampler_view_sample
        self.sampler_frame_sample = sampler_frame_sample
        self.select_samples()

    def load_source_params(self):
        # Perform view selection first
        view_inds = self.frame_inds if self.closest_using_t else self.view_inds
        view_inds = torch.arange(0, len(view_inds))
        if len(self.src_view_sample) != 3: view_inds = view_inds[self.src_view_sample]  # this is a list of indices
        else: view_inds = view_inds[self.src_view_sample[0]:self.src_view_sample[1]:self.src_view_sample[2]]  # begin, start, end
        self.src_view_inds = view_inds
        if len(view_inds) == 1: view_inds = [view_inds]  # MARK: pytorch indexing bug, when length is 1, will reduce a dim

        # For getting the actual data (renaming w2c and K)
        # See easyvolcap/dataloaders/datasets/image_based_inference_dataset
        if self.closest_using_t:  # this checks whether the view selection is performed on the frame or view dim
            self.src_ixts = self.Ks[:, view_inds]  # N, L, 4, 4
            self.src_exts = affine_padding(self.w2cs[:, view_inds])  # N, L, 4, 4
            self.src_ixts = self.src_ixts.permute(1, 0, 2, 3)  # L, N, 4, 4 # MARK: transpose
            self.src_exts = self.src_exts.permute(1, 0, 2, 3)  # L, N, 4, 4 # MARK: transpose
        else:
            self.src_ixts = self.Ks[view_inds]  # N, L, 4, 4
            self.src_exts = affine_padding(self.w2cs[view_inds])  # N, L, 4, 4

    def load_source_indices(self):
        # Get the target views and source views
        tar_c2ws = self.c2ws.permute(1, 0, 2, 3) if self.closest_using_t else self.c2ws  # MARK: transpose
        src_c2ws = affine_inverse(self.src_exts)

        # Load cached source indices and similarities directly if available
        if self.use_cached_srcs and exists(join(self.data_root, self.source_dir, self.src_inds_file)):
            # Load source indices and similarities
            self.src_inds = torch.as_tensor(np.load(join(self.data_root, self.source_dir, self.src_inds_file)), dtype=torch.int64)  # (V, V)
            self.src_sims = torch.as_tensor(np.load(join(self.data_root, self.source_dir, self.src_sims_file)), dtype=torch.float)  # (V, V)
            # Perform view selection
            self.src_inds = self.src_inds[self.view_inds]  # (Vt, V)
            self.src_sims = self.src_sims[self.view_inds]  # (Vt, V)
            # Perform source view selection
            inds = (self.src_inds[..., None] == self.src_view_inds).nonzero()[:, :2]
            self.src_inds = self.src_inds[tuple(inds.t())].view(self.n_views, -1)  # (Vt, Vs)
            self.src_sims = self.src_sims[tuple(inds.t())].view(self.n_views, -1)  # (Vt, Vs)
            # Repeat for the number of latents
            self.src_inds = self.src_inds[:, :, None].repeat(1, 1, self.n_latents)  # (Vt, Vs, L)
            self.src_sims = self.src_sims[:, :, None].repeat(1, 1, self.n_latents)  # (Vt, Vs, L)
        else:
            if self.source_type == Sourcing.DISTANCE:
                self.src_sims, self.src_inds = compute_camera_similarity(tar_c2ws, src_c2ws)  # similarity to source views # Target, Source, Latent
            elif self.source_type == Sourcing.ZIGZAG:
                self.src_sims, self.src_inds = compute_camera_zigzag_similarity(tar_c2ws, src_c2ws)  # similarity to source views # Target, Source, Latent
            else:
                raise NotImplementedError

    def load_paths(self):
        # Load image related stuff for reading from disk later
        # If number of images in folder does not match, here we'll get an error
        ims = [[join(self.data_root, self.images_dir, cam, self.ims_pattern.format(frame=i)) for i in range(self.n_frames_total)] for cam in self.camera_names]
        if not exists(ims[0][0]):
            ims = [[i.replace('.' + self.ims_pattern.split('.')[-1], '.JPG') for i in im] for im in ims]
        if not exists(ims[0][0]):
            ims = [[i.replace('.JPG', '.png') for i in im] for im in ims]
        if not exists(ims[0][0]):
            ims = [[i.replace('.png', '.PNG') for i in im] for im in ims]
        if not exists(ims[0][0]):
            ims = [sorted(glob(join(self.data_root, self.images_dir, cam, '*')))[:self.n_frames_total] for cam in self.camera_names]
        ims = [np.asarray(ims[i])[:min([len(i) for i in ims])] for i in range(len(ims))]  # deal with the fact that some weird dataset has different number of images
        self.ims = np.asarray(ims)  # V, N
        self.ims_dir = join(*split(dirname(self.ims[0, 0]))[:-1])  # logging only

        # TypeError: can't convert np.ndarray of type numpy.str_. The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint8, and bool.
        # MARK: Names stored as np.ndarray
        inds = np.arange(self.ims.shape[-1])
        if len(self.frame_sample) != 3: inds = inds[self.frame_sample]
        else: inds = inds[self.frame_sample[0]:self.frame_sample[1]:self.frame_sample[2]]
        self.ims = self.ims[..., inds]  # these paths are later used for reading images from disk

        # Mask path preparation
        if self.use_masks:
            self.mks = np.asarray([im.replace(self.images_dir, self.masks_dir) for im in self.ims.ravel()]).reshape(self.ims.shape)
            if not exists(self.mks[0, 0]):
                self.mks = np.asarray([mk.replace('.png', '.jpg') for mk in self.mks.ravel()]).reshape(self.mks.shape)
            if not exists(self.mks[0, 0]):
                self.mks = np.asarray([mk.replace('.jpg', '.png') for mk in self.mks.ravel()]).reshape(self.mks.shape)
            if not exists(self.mks[0, 0]):  # Two types of commonly used mask directories
                self.mks = np.asarray([mk.replace(self.masks_dir, 'masks') for mk in self.mks.ravel()]).reshape(self.mks.shape)
            if not exists(self.mks[0, 0]):
                self.mks = np.asarray([mk.replace('masks', 'mask') for mk in self.mks.ravel()]).reshape(self.mks.shape)
            if not exists(self.mks[0, 0]):
                self.mks = np.asarray([mk.replace('masks', 'msk') for mk in self.mks.ravel()]).reshape(self.mks.shape)
            self.mks_dir = join(*split(dirname(self.mks[0, 0]))[:-1])

        # Depth image path preparation
        if self.use_depths:
            self.dps = np.asarray([im.replace(self.images_dir, self.depths_dir).replace('.jpg', '.exr').replace('.png', '.exr') for im in self.ims.ravel()]).reshape(self.ims.shape)
            if not exists(self.dps[0, 0]):
                self.dps = np.asarray([dp.replace('.exr', 'exr') for dp in self.dps.ravel()]).reshape(self.dps.shape)
            self.dps_dir = join(*split(dirname(self.dps[0, 0]))[:-1])  # logging only

        # Normal image path preparation
        if self.use_normals:
            self.nms = np.asarray([im.replace(self.images_dir, self.normals_dir) for im in self.ims.ravel()]).reshape(self.ims.shape)
            if not exists(self.nms[0, 0]):
                self.nms = np.asarray([nm.replace('.png', '.jpg') for nm in self.nms.ravel()]).reshape(self.nms.shape)
            if not exists(self.nms[0, 0]):
                self.nms = np.asarray([nm.replace('.jpg', '.png') for nm in self.nms.ravel()]).reshape(self.nms.shape)
            self.nms_dir = join(*split(dirname(self.nms[0, 0]))[:-1])  # logging only

    def load_bytes(self):
        # Camera distortions are only applied on the ground truth image, the rendering model does not include these
        # And unlike intrinsic parameters, it has no direct dependency on the size of the loaded image, thus we directly process them here
        dist_mask = torch.as_tensor(self.dist_mask)
        self.Ds = self.Ds.view(*self.Ds.shape[:2], 5) * dist_mask  # some of the distortion parameters might need some manual massaging

        # Need to convert to a tight data structure for access
        ori_Ks = self.Ks
        ori_Ds = self.Ds
        # msk_Ds = ori_Ds.clone()  # this is a DNA-Rendering special
        # msk_Ds[..., -1] = 0.0  # only use the first 4 distortion parameters for mask undistortion
        # msk_Ds = torch.zeros_like(ori_Ds) # avoid bad distortion params
        ratio = self.imsize_overwrite if self.imsize_overwrite[0] > 0 else self.ratio  # maybe force size, or maybe use ratio to resize
        if self.use_masks:
            self.mks_bytes, self.Ks, self.Hs, self.Ws = \
                load_resize_undist_ims_bytes(self.mks, ori_Ks.numpy(), ori_Ds.numpy(), ratio, self.center_crop_size,
                                             f'Loading mask bytes for {blue(self.mks_dir)} {magenta(self.split.name)}',
                                             decode_flag=cv2.IMREAD_GRAYSCALE, dist_opt_K=self.dist_opt_K, encode_ext=self.encode_ext)  # will for a grayscale read from bytes
            self.Ks = torch.as_tensor(self.Ks)
            self.Hs = torch.as_tensor(self.Hs)
            self.Ws = torch.as_tensor(self.Ws)

        # Maybe load depth images here, using EXR
        if self.use_depths:
            self.dps_bytes, self.Ks, self.Hs, self.Ws = \
                load_resize_undist_ims_bytes(self.dps, ori_Ks.numpy(), ori_Ds.numpy(), ratio, self.center_crop_size,
                                             f'Loading dpts bytes for {blue(self.dps_dir)} {magenta(self.split.name)}',
                                             decode_flag=cv2.IMREAD_UNCHANGED, dist_opt_K=self.dist_opt_K, encode_ext='.exr')  # will for a grayscale read from bytes

        # Maybe load normal images here
        if self.use_normals:
            self.nms_bytes, self.Ks, self.Hs, self.Ws = \
                load_resize_undist_ims_bytes(self.nms, ori_Ks.numpy(), ori_Ds.numpy(), ratio, self.center_crop_size,
                                             f'Loading norm bytes for {blue(self.nms_dir)} {magenta(self.split.name)}',
                                             dist_opt_K=self.dist_opt_K, encode_ext=self.encode_ext)

        # Image pre cacheing (from disk to memory)
        self.ims_bytes, self.Ks, self.Hs, self.Ws = \
            load_resize_undist_ims_bytes(self.ims, ori_Ks.numpy(), ori_Ds.numpy(), ratio, self.center_crop_size,
                                         f'Loading imgs bytes for {blue(self.ims_dir)} {magenta(self.split.name)}',
                                         dist_opt_K=self.dist_opt_K, encode_ext=self.encode_ext)

        self.Ks = torch.as_tensor(self.Ks)
        self.Hs = torch.as_tensor(self.Hs)
        self.Ws = torch.as_tensor(self.Ws)

        # Precrop image to bytes
        if self.immask_crop:  # a little bit wasteful but acceptable for now
            # TODO: Assert that mask crop is always more aggressive than bounds crop (intersection of interested area)
            self.orig_hs, self.orig_ws = self.Hs, self.Ws
            bounds = [self.get_bounds(i) for i in range(self.n_latents)]  # N, 2, 3
            bounds = torch.stack(bounds)[None].repeat(self.n_views, 1, 1, 1)  # V, N, 2, 3

            if hasattr(self, 'dps_bytes'): self.dps_bytes, mks_bytes, Ks, Hs, Ws, crop_xs, crop_ys = \
                decode_crop_fill_ims_bytes(self.dps_bytes, self.mks_bytes, self.Ks.numpy(), self.Rs.numpy(), self.Ts.numpy(), bounds.numpy(), f'Cropping msks dpts for {blue(self.data_root)} {magenta(self.split.name)}', encode_ext=['.exr', self.encode_ext])

            if hasattr(self, 'nms_bytes'): self.nms_bytes, mks_bytes, Ks, Hs, Ws, crop_xs, crop_ys = \
                decode_crop_fill_ims_bytes(self.nms_bytes, self.mks_bytes, self.Ks.numpy(), self.Rs.numpy(), self.Ts.numpy(), bounds.numpy(), f'Cropping msks nrms for {blue(self.data_root)} {magenta(self.split.name)}', encode_ext=self.encode_ext)

            self.ims_bytes, self.mks_bytes, self.Ks, self.Hs, self.Ws, self.crop_xs, self.crop_ys = \
                decode_crop_fill_ims_bytes(self.ims_bytes, self.mks_bytes, self.Ks.numpy(), self.Rs.numpy(), self.Ts.numpy(), bounds.numpy(), f'Cropping msks imgs for {blue(self.data_root)} {magenta(self.split.name)}', encode_ext=self.encode_ext)

            self.corp_xs = torch.as_tensor(self.crop_xs)
            self.corp_ys = torch.as_tensor(self.crop_ys)
            self.Ks = torch.as_tensor(self.Ks)
            self.Hs = torch.as_tensor(self.Hs)
            self.Ws = torch.as_tensor(self.Ws)

        # Only fill the background regions
        if not self.immask_crop and self.immask_fill:  # a little bit wasteful but acceptable for now
            self.ims_bytes = decode_fill_ims_bytes(self.ims_bytes, self.mks_bytes, f'Filling msks imgs for {blue(self.data_root)} {magenta(self.split.name)}', encode_ext=self.encode_ext)
            if hasattr(self, 'dps_bytes'): self.dps_bytes = decode_fill_ims_bytes(self.dps_bytes, self.mks_bytes, f'Filling dpts imgs for {blue(self.data_root)} {magenta(self.split.name)}', encode_ext='.exr')
            if hasattr(self, 'nms_bytes'): self.nms_bytes = decode_fill_ims_bytes(self.nms_bytes, self.mks_bytes, f'Filling norm imgs for {blue(self.data_root)} {magenta(self.split.name)}', encode_ext=self.encode_ext)

        # To make memory access faster, store raw floats in memory
        if self.cache_raw:
            self.ims_bytes = to_tensor([load_image_from_bytes(x, normalize=True) for x in tqdm(self.ims_bytes, desc=f'Caching imgs for {blue(self.data_root)} {magenta(self.split.name)}')])  # High mem usage
            if hasattr(self, 'mks_bytes'): self.mks_bytes = to_tensor([load_image_from_bytes(x, normalize=True) for x in tqdm(self.mks_bytes, desc=f'Caching mks for {blue(self.data_root)} {magenta(self.split.name)}')])
            if hasattr(self, 'dps_bytes'): self.dps_bytes = to_tensor([load_image_from_bytes(x, normalize=False) for x in tqdm(self.dps_bytes, desc=f'Caching dps for {blue(self.data_root)} {magenta(self.split.name)}')])
            if hasattr(self, 'nms_bytes'): self.nms_bytes = to_tensor([load_image_from_bytes(x, normalize=True) for x in tqdm(self.nms_bytes, desc=f'Caching nms for {blue(self.data_root)} {magenta(self.split.name)}')])
        else:
            # Avoid splitting memory for bytes objects
            self.ims_bytes = UnstructuredTensors(self.ims_bytes)
            if hasattr(self, 'mks_bytes'): self.mks_bytes = UnstructuredTensors(self.mks_bytes)
            if hasattr(self, 'dps_bytes'): self.dps_bytes = UnstructuredTensors(self.dps_bytes)
            if hasattr(self, 'nms_bytes'): self.nms_bytes = UnstructuredTensors(self.nms_bytes)

    def load_cameras(self):
        # Load camera related stuff like image list and intri, extri.
        # Determine whether it is a monocular dataset or multiview dataset based on the existence of root `extri.yml` or `intri.yml`
        # Multiview dataset loading, need to expand, will have redundant information
        if exists(join(self.data_root, self.intri_file)) and exists(join(self.data_root, self.extri_file)):
            self.cameras = read_camera(join(self.data_root, self.intri_file), join(self.data_root, self.extri_file))
            self.camera_names = np.asarray(sorted(list(self.cameras.keys())))  # NOTE: sorting camera names
            self.cameras = dotdict({k: [self.cameras[k] for i in range(self.n_frames_total)] for k in self.camera_names})
            # TODO: Handle avg processing

        # Monocular dataset loading, each camera has a separate folder
        elif exists(join(self.data_root, self.cameras_dir)):
            self.camera_names = np.asarray(sorted(os.listdir(join(self.data_root, self.cameras_dir))))  # NOTE: sorting here is very important!
            self.cameras = dotdict({
                k: [v[1] for v in sorted(
                    read_camera(join(self.data_root, self.cameras_dir, k, self.intri_file),
                                join(self.data_root, self.cameras_dir, k, self.extri_file)).items()
                )] for k in self.camera_names
            })
            # TODO: Handle avg export and loading for such monocular dataset
        else:
            log(red('Could not find camera information in the dataset, check your dataset configuration'))
            log(red('If you want to render the model without loading anything from the dataset:'))
            log(red('Try appending val_dataloader_cfg.dataset_cfg.type=NoopDataset to your command or add the `configs/specs/turbom.yaml` to your `-c` parameter'))
            raise NotImplementedError(f'Could not find {{{self.intri_file},{self.extri_file}}} or {self.cameras_dir} directory in {self.data_root}, check your dataset configuration or use NoopDataset')

        # Expectation:
        # self.camera_names: a list containing all camera names
        # self.cameras: a mapping from camera names to a list of camera objects
        # (every element in list is an actual camera for that particular view and frame)
        # NOTE: ALWAYS, ALWAYS, SORT CAMERA NAMES.
        self.Hs = torch.as_tensor([[cam.H for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F
        self.Ws = torch.as_tensor([[cam.W for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F
        self.Ks = torch.as_tensor([[cam.K for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F, 3, 3
        self.Rs = torch.as_tensor([[cam.R for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F, 3, 3
        self.Ts = torch.as_tensor([[cam.T for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F, 3, 1
        self.Ds = torch.as_tensor([[cam.D for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F, 1, 5
        self.ts = torch.as_tensor([[cam.t for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F
        self.ns = torch.as_tensor([[cam.n for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F
        self.fs = torch.as_tensor([[cam.f for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F
        self.Cs = -self.Rs.mT @ self.Ts  # V, F, 3, 1
        self.w2cs = torch.cat([self.Rs, self.Ts], dim=-1)  # V, F, 3, 4
        self.c2ws = affine_inverse(self.w2cs)  # V, F, 3, 4
        self.ns, self.fs = monotonic_near_far(self.ns, self.fs, torch.as_tensor(self.near, dtype=torch.float), torch.as_tensor(self.far, dtype=torch.float))
        self.near, self.far = max(self.near, self.ns.min()), min(self.far, self.fs.max())

        # Move cameras to the center of the frame (!: intrusive)
        if self.use_aligned_cameras:
            self.align_cameras()

    def align_cameras(self):
        sh = self.c2ws.shape  # V, F, 3, 4
        self.c2ws = self.c2ws.view((-1,) + sh[-2:])  # V*F, 3, 4

        if self.avg_using_all:
            stride = max(len(self.c2ws) // self.avg_max_count, 1)
            inds = torch.arange(len(self.c2ws))[::stride][:self.avg_max_count]
            c2w_avg = as_torch_func(average_c2ws)(self.c2ws[inds])  # V*F, 3, 4, # !: HEAVY
        else:
            c2w_avg = as_torch_func(average_c2ws)(self.c2ws.view(sh)[:, 0])  # V, 3, 4
        self.c2w_avg = c2w_avg

        self.c2ws = (affine_inverse(affine_padding(self.c2w_avg))[None] @ affine_padding(self.c2ws))[..., :3, :]  # 1, 4, 4 @ V*F, 4, 4 -> V*F, 3, 4
        self.w2cs = affine_inverse(self.c2ws)  # V*F, 3, 4
        self.c2ws = self.c2ws.view(sh)
        self.w2cs = self.w2cs.view(sh)

        self.Rs = self.w2cs[..., :-1]
        self.Ts = self.w2cs[..., -1:]
        self.Cs = self.c2ws[..., -1:]  # updated camera center

    def select_cameras(self):
        # Only retrain needed
        # Perform view selection first
        view_inds = torch.arange(self.Ks.shape[0])
        if len(self.view_sample) != 3: view_inds = view_inds[self.view_sample]  # this is a list of indices
        else: view_inds = view_inds[self.view_sample[0]:self.view_sample[1]:self.view_sample[2]]  # begin, start, end
        self.view_inds = view_inds
        if len(view_inds) == 1: view_inds = [view_inds]  # MARK: pytorch indexing bug, when length is 1, will reduce a dim

        # Perform frame selection next
        frame_inds = torch.arange(self.Ks.shape[1])
        if len(self.frame_sample) != 3: frame_inds = frame_inds[self.frame_sample]
        else: frame_inds = frame_inds[self.frame_sample[0]:self.frame_sample[1]:self.frame_sample[2]]
        self.frame_inds = frame_inds  # used by `load_smpls()`
        if len(frame_inds) == 1: frame_inds = [frame_inds]  # MARK: pytorch indexing bug, when length is 1, will reduce a dim

        # NOTE: if view_inds == [0,] in monocular dataset or whatever case, type(`self.camera_names[view_inds]`) == str, not a list of str
        self.camera_names = np.asarray([self.camera_names[view] for view in view_inds])  # this is what the b, e, s means
        self.cameras = dotdict({k: [self.cameras[k][int(i)] for i in frame_inds] for k in self.camera_names})  # reloading
        self.Hs = self.Hs[view_inds][:, frame_inds]
        self.Ws = self.Ws[view_inds][:, frame_inds]
        self.Ks = self.Ks[view_inds][:, frame_inds]
        self.Rs = self.Rs[view_inds][:, frame_inds]
        self.Ts = self.Ts[view_inds][:, frame_inds]
        self.Ds = self.Ds[view_inds][:, frame_inds]
        self.ts = self.ts[view_inds][:, frame_inds]
        self.Cs = self.Cs[view_inds][:, frame_inds]
        self.w2cs = self.w2cs[view_inds][:, frame_inds]
        self.c2ws = self.c2ws[view_inds][:, frame_inds]

    def select_samples(self):
        # Perform sampler view selection first
        sampler_view_inds = torch.arange(self.Ks.shape[0])
        if len(self.sampler_view_sample) != 3: sampler_view_inds = sampler_view_inds[self.sampler_view_sample]  # this is a list of indices
        else: sampler_view_inds = sampler_view_inds[self.sampler_view_sample[0]:self.sampler_view_sample[1]:self.sampler_view_sample[2]]  # begin, start, end
        self.sampler_view_inds = sampler_view_inds

        # Perform sampler frame selection next
        sampler_frame_inds = torch.arange(self.Ks.shape[1])
        if len(self.sampler_frame_sample) != 3: sampler_frame_inds = sampler_frame_inds[self.sampler_frame_sample]
        else: sampler_frame_inds = sampler_frame_inds[self.sampler_frame_sample[0]:self.sampler_frame_sample[1]:self.sampler_frame_sample[2]]
        self.sampler_frame_inds = sampler_frame_inds

    # def get_indices(self, index):
    #     # These indices are relative to the processed dataset
    #     view_index, latent_index = index // self.n_latents, index % self.n_latents

    #     if len(self.view_sample) != 3: camera_index = self.view_sample[view_index]
    #     else: camera_index = view_index * self.view_sample[2] + self.view_sample[0]

    #     if len(self.frame_sample) != 3: frame_index = self.frame_sample[latent_index]
    #     else: frame_index = latent_index * self.frame_sample[2] + self.frame_sample[0]

    #     return view_index, latent_index, camera_index, frame_index

    # NOTE: everything beginning with get are utilities for __getitem__
    # NOTE: coding convension are preceded with "NOTE"
    def get_indices(self, index):
        # These indices are relative to the data sampler
        sampler_view_index, sampler_latent_index = index // self.sampler_n_latents, index % self.sampler_n_latents

        # Use the sampler indices to get the latent indices
        view_index = self.sampler_view_inds[sampler_view_index]
        latent_index = self.sampler_frame_inds[sampler_latent_index]

        # Use the latent indices to get the actual physical indices
        if len(self.view_sample) != 3: camera_index = self.view_sample[view_index]
        else: camera_index = view_index * self.view_sample[2] + self.view_sample[0]
        if len(self.frame_sample) != 3: frame_index = self.frame_sample[latent_index]
        else: frame_index = latent_index * self.frame_sample[2] + self.frame_sample[0]

        return view_index, latent_index, camera_index, frame_index

    def get_image_bytes(self, view_index: int, latent_index: int):
        im_bytes = self.ims_bytes[view_index * self.n_latents + latent_index]  # MARK: no fancy indexing

        if self.use_masks:
            mk_bytes = self.mks_bytes[view_index * self.n_latents + latent_index]
        else:
            mk_bytes = None

        if self.use_depths:
            dp_bytes = self.dps_bytes[view_index * self.n_latents + latent_index]
        else:
            dp_bytes = None

        if self.use_normals:
            nm_bytes = self.nms_bytes[view_index * self.n_latents + latent_index]
        else:
            nm_bytes = None

        return im_bytes, mk_bytes, dp_bytes, nm_bytes

    def get_image(self, view_index: int, latent_index: int):
        # Load bytes (rgb, msk, dpt, norm)
        im_bytes, mk_bytes, dp_bytes, nm_bytes = self.get_image_bytes(view_index, latent_index)
        rgb, msk, dpt, norm = None, None, None, None

        # Load image from bytes
        if self.cache_raw:
            rgb = torch.as_tensor(im_bytes)
        else:
            rgb = torch.as_tensor(load_image_from_bytes(im_bytes, normalize=True))  # 4-5ms for 400 * 592 jpeg, sooo slow

        # Load mask from bytes
        if mk_bytes is not None:
            if self.cache_raw:
                msk = torch.as_tensor(mk_bytes)
            else:
                msk = torch.as_tensor(load_image_from_bytes(mk_bytes, normalize=True)[..., :1])
        else:
            msk = torch.ones_like(rgb[..., -1:])

        # Load depth from bytes
        if dp_bytes is not None:
            if self.cache_raw:
                dpt = torch.as_tensor(dp_bytes)
            else:
                dpt = torch.as_tensor(load_image_from_bytes(dp_bytes, normalize=False)[..., :1])  # readin as is

        # Load normal from bytes
        if nm_bytes is not None:
            if self.cache_raw:
                norm = torch.as_tensor(nm_bytes)
            else:
                norm = torch.as_tensor(load_image_from_bytes(nm_bytes, normalize=True))  # readin as is

        return rgb, msk, dpt, norm

    def load_image(self, view_index: int, latent_index: int):
        # This function is used for loading image, depth, ... for a specific sample from disk
        # It will be invoked by __getitem__ when `self.skip_loading_images` is True to avoid OOM since we may have a large dataset
        ratio = self.imsize_overwrite if self.imsize_overwrite[0] > 0 else self.ratio  # maybe force size, or maybe use ratio to resize
        rgb, msk, dpt, norm = None, None, None, None

        # Maybe load mask here
        if self.use_masks:
            mk_bytes, K, H, W = \
                load_resize_undist_im_bytes(self.mks[view_index, latent_index], self.Ks[view_index, latent_index].numpy(),
                                            self.Ds[view_index, latent_index].numpy(), ratio, self.center_crop_size,
                                            decode_flag=cv2.IMREAD_GRAYSCALE, dist_opt_K=self.dist_opt_K, encode_ext=self.encode_ext)

        # Maybe load depth images here, using EXR
        if self.use_depths:
            dp_bytes, K, H, W = \
                load_resize_undist_im_bytes(self.dps[view_index, latent_index], self.Ks[view_index, latent_index].numpy(),
                                            self.Ds[view_index, latent_index].numpy(), ratio, self.center_crop_size,
                                            decode_flag=cv2.IMREAD_UNCHANGED, dist_opt_K=self.dist_opt_K, encode_ext='.exr')

        # Maybe load normal images here
        if self.use_normals:
            nm_bytes, K, H, W = \
                load_resize_undist_im_bytes(self.nms[view_index, latent_index], self.Ks[view_index, latent_index].numpy(),
                                            self.Ds[view_index, latent_index].numpy(), ratio, self.center_crop_size,
                                            dist_opt_K=self.dist_opt_K, encode_ext=self.encode_ext)

        # Load image from bytes
        im_bytes, K, H, W = \
            load_resize_undist_im_bytes(self.ims[view_index, latent_index], self.Ks[view_index, latent_index].numpy(),
                                        self.Ds[view_index, latent_index].numpy(), ratio, self.center_crop_size,
                                        dist_opt_K=self.dist_opt_K, encode_ext=self.encode_ext)

        # Convert to tensor
        rgb = torch.as_tensor(load_image_from_bytes(im_bytes, normalize=True))  # H, W, 3

        if self.use_masks:
            msk = torch.as_tensor(load_image_from_bytes(mk_bytes, normalize=True)[..., :1])  # H, W, 1
        else:
            msk = torch.ones_like(rgb[..., -1:])  # H, W, 1

        if self.use_depths:
            dpt = torch.as_tensor(load_image_from_bytes(dp_bytes, normalize=False)[..., :1])  # H, W, 1

        if self.use_normals:
            norm = torch.as_tensor(load_image_from_bytes(nm_bytes, normalize=True))  # H, W, 3

        return rgb, msk, dpt, norm, torch.as_tensor(K), torch.as_tensor(H), torch.as_tensor(W)

    def get_camera_params(self, view_index, latent_index):
        w2c, c2w = self.w2cs[view_index][latent_index], self.c2ws[view_index][latent_index]
        R, T = self.Rs[view_index][latent_index], self.Ts[view_index][latent_index]  # 4, 4; 3, 3; 3, 1; 5, 1
        n, f = self.ns[view_index][latent_index], self.fs[view_index][latent_index]  # 1, 1
        t = self.ts[view_index][latent_index]  # 1

        # These might be invalid
        H, W, K = self.Hs[view_index][latent_index], self.Ws[view_index][latent_index], self.Ks[view_index][latent_index]
        return w2c, c2w, R, T, H, W, K, n, f, t

    def get_bounds(self, latent_index):
        bounds = self.bounds.clone()  # always copy before inplace operation
        bounds[0] = torch.maximum(bounds[0], self.bounds[0])
        bounds[1] = torch.minimum(bounds[1], self.bounds[1])
        return bounds

    @property
    def n_views(self): return len(self.cameras)

    @property
    def n_latents(self): return len(next(iter(self.cameras.values())))  # short for timestamp

    @property
    def sampler_n_views(self): return len(self.sampler_view_inds)

    @property
    def sampler_n_latents(self): return len(self.sampler_frame_inds)

    @property
    def frame_min(self): return self.frame_sample[0] if len(self.frame_sample) == 3 else min(self.frame_sample)

    @property
    def frame_max(self):
        middle = (self.frame_sample[1] if self.frame_sample[1] else self.n_frames_total) - 1  # None -> all frames are loaded
        return middle if len(self.frame_sample) == 3 else max(self.frame_sample)

    @property
    def frame_int(self): return self.frame_sample[2] if len(self.frame_sample) == 3 else -1  # error out if you call this when giving specific frames

    @property
    def frame_range(self):
        return np.clip(self.frame_max - self.frame_min, 1, None)

    @property
    def view_min(self): return self.view_sample[0] if len(self.view_sample) == 3 else min(self.view_sample)

    @property
    def view_max(self):
        middle = (self.view_sample[1] if self.view_sample[1] else self.n_view_total) - 1  # None -> all frames are loaded
        return middle if len(self.view_sample) == 3 else max(self.view_sample)

    @property
    def view_int(self): return self.view_sample[2] if len(self.view_sample) == 3 else -1  # error out if you call this when giving specific frames

    @property
    def view_range(self):
        return np.clip(self.view_max - self.view_min, 1, None)

    def t_to_frame(self, t):
        return int(t * (self.frame_max - self.frame_min) + self.frame_min + 1e-5)  # avoid out of bounds

    def frame_to_t(self, frame_index):
        return (frame_index - self.frame_min) / self.frame_range  # avoid division by 0

    def frame_to_latent(self, frame_index):
        return int((frame_index - self.frame_min) / self.frame_int + 1e-5)  # avoid out of bounds

    def camera_to_v(self, camera_index):
        return (camera_index - self.view_min) / self.view_range  # avoid division by 0

    def v_to_camera(self, v):
        return int(v * (self.view_max - self.view_min) + self.view_min + 1e-5)  # avoid out of bounds

    def camera_to_view(self, camera_index):
        return int((camera_index - self.view_min) / self.view_int + 1e-5)  # avoid out of bounds

    def get_metadata(self, index):
        view_index, latent_index, camera_index, frame_index = self.get_indices(index)
        w2c, c2w, R, T, H, W, K, n, f, t = self.get_camera_params(view_index, latent_index)

        # NOTE: everything meta in the dataset are ignored when copying to cuda (avoiding syncing)
        meta = dotdict()  # camera parameters
        meta.H, meta.W = H, W
        meta.K, meta.R, meta.T = K, R, T
        meta.n, meta.f = n, f  # canonical
        meta.range = f - n  # canonical
        meta.scale = self.scale  # original -> canonical
        meta.w2c, meta.c2w = w2c, c2w
        meta.view_index, meta.latent_index, meta.camera_index, meta.frame_index = view_index, latent_index, camera_index, frame_index
        meta.t = (t / self.duration) if self.use_loaded_time else self.frame_to_t(frame_index)  # for duration of 1.0, this is a no-op
        meta.t = torch.as_tensor(meta.t, dtype=torch.float)  # the dataset provided time or the time fraction
        meta.v = self.camera_to_v(camera_index)
        meta.v = torch.as_tensor(meta.v, dtype=torch.float)  # the time fraction
        if self.use_aligned_cameras: meta.c2w_avg = self.c2w_avg  # MARK: store the aligned cameras here

        # Other inputs
        meta.bounds = self.get_bounds(latent_index)

        output = dotdict()
        output.update(meta)  # will also store a copy of these metadata on GPU
        output.meta = dotdict()  # this is the first time that this metadata is created in the batch
        output.meta.update(meta)

        # Maybe crop intrinsics
        if self.imbound_crop:
            self.crop_ixts_bounds(output)  # only crop target ixts

        return output

    @staticmethod
    def scale_ixts(output: dotdict, ratio: float):
        orig_h, orig_w = output.H, output.W
        new_h, new_w = int(orig_h * ratio), int(orig_w * ratio)
        ratio_h, ratio_w = new_h / orig_h, new_w / orig_w
        K = output.K.clone()
        K[0:1] *= ratio_w
        K[1:2] *= ratio_h
        meta = dotdict()
        meta.K = K
        meta.tar_ixt = K
        meta.H = torch.as_tensor(new_h)
        meta.W = torch.as_tensor(new_w)
        if 'orig_h' in output:
            meta.crop_x = torch.as_tensor(int(output.crop_x * ratio))
            meta.crop_y = torch.as_tensor(int(output.crop_y * ratio))  # TODO: this is messy
            meta.orig_h = torch.as_tensor(int(output.orig_h * ratio))
            meta.orig_w = torch.as_tensor(int(output.orig_w * ratio))
            # Now the K corresponds to crop_x * self.render_ratio instead of int(output.crop_x * self.render_ratio)
            # We should fix that: no resizing, only a fractional movement here
            meta.K[..., :2, -1] -= torch.as_tensor([output.crop_x * ratio - meta.crop_x,  # only dealing with the cropped fraction
                                                    output.crop_y * ratio - meta.crop_y,  # only dealing with the cropped fraction
                                                    ],
                                                   device=output.bounds.device)  # crop K
        output.update(meta)
        output.meta.update(meta)
        return output

    @staticmethod
    def crop_ixts(output: dotdict, x, y, w, h):
        """
        Crops target intrinsics using a xywh 
        """
        K = output.K.clone()
        K[..., :2, -1] -= torch.as_tensor([x, y], device=output.bounds.device)  # crop K

        output.K = K
        output.tar_ixt = K

        meta = dotdict()
        meta.K = K
        meta.H = torch.as_tensor(h, device=output.bounds.device)
        meta.W = torch.as_tensor(w, device=output.bounds.device)
        if 'crop_x' in output.meta:
            meta.crop_x = torch.as_tensor(x + output.meta.crop_x, device=output.bounds.device)
            meta.crop_y = torch.as_tensor(y + output.meta.crop_y, device=output.bounds.device)
        else:
            meta.crop_x = torch.as_tensor(x, device=output.bounds.device)
            meta.crop_y = torch.as_tensor(y, device=output.bounds.device)
            meta.orig_w = output.W  # original size before update
            meta.orig_h = output.H  # original size before update
        output.update(meta)
        output.meta.update(meta)

        return output

    def get_ibr_metadata(self, index):
        if isinstance(index, dotdict): index, n_srcs = index.index, index.n_srcs
        else: n_srcs = random.choices(self.n_srcs_list, self.n_srcs_prob)[0]

        # Load target view related stuff
        output = MultiViewDataset.get_metadata(self, index)  # target view camera matrices

        # Load target view enerf specific stuff
        # There's this strange convension in ENeRF: ext: 4x4 homogeneous matrix, c2w: 3x4 reduced mat
        output.tar_ext = affine_padding(output.w2c)  # ? renaming things
        output.tar_ixt = output.K  # 3, 3, avoid being modified later

        # Load source view related stuff
        if self.closest_using_t:  # selecting closest view along temporal dimension # MARK: transpose
            target_index = output.latent_index
            extra_index = output.view_index
        else:
            target_index = output.view_index
            extra_index = output.latent_index

        # For training, maybe sample the original image
        # remove_gt = 1 if random.random() > self.append_gt_prob else 0  # training and random -> exclude gt
        # NOTE: for stereo or mvs depth estimation, target view is target view, it should never be one of its source views
        remove_gt = 1
        random_ap = self.extra_src_pool  # training -> randomly sample more
        src_inds = self.src_inds[target_index, remove_gt:remove_gt + n_srcs + random_ap, extra_index]  # excluding the target view, 5 inds
        if random_ap: src_inds = torch.as_tensor(random.sample(src_inds.numpy().tolist(), n_srcs))  # S (2, 4)

        output.t_inds = extra_index
        output.meta.t_inds = extra_index
        output.src_exts = self.src_exts[src_inds, extra_index]  # S, 4, 4
        output.src_ixts = self.src_ixts[src_inds, extra_index]  # S, 3, 3
        output.meta.src_exts = output.src_exts
        output.meta.src_ixts = output.src_ixts

        # Other bookkeepings
        src_inds = self.src_view_inds.gather(-1, src_inds)  # S, -> T, S, L -> T, S, L
        output.src_inds = src_inds  # as tensors
        output.meta.src_inds = src_inds  # as tensors

        source_index = src_inds.detach().cpu().numpy().tolist()
        if self.closest_using_t:  # selecting closest view along temporal dimension # MARK: transpose
            latent_index = source_index
            view_index = extra_index
        else:
            latent_index = extra_index
            view_index = source_index

        output = self.get_sources(latent_index, view_index, output)

        return output

    def get_sources(self, latent_index: Union[List[int], int], view_index: Union[List[int], int], output: dotdict):
        # Load all the source images, depths, etc
        if not self.skip_loading_images:
            rgb, msk, dpt, norm = zip(*parallel_execution(view_index, latent_index, action=self.get_image, sequential=True))
        else:
            rgb, msk, dpt, norm, K, H, W = zip(*parallel_execution(view_index, latent_index, action=self.load_image, sequential=True))
            output.src_ixts = torch.stack(K)  # S, 3, 3
            output.meta.src_ixts = output.src_ixts

        # Process the images
        output.src_inps = torch.stack([i.permute(2, 0, 1) for i in rgb], dim=0)  # for data locality # S, H, W, 3 -> S, 3, H, W
        output.src_msks = torch.stack([i.permute(2, 0, 1) for i in msk], dim=0)  # for data locality # S, H, W, 1 -> S, 1, H, W
        if dpt[0] is not None: output.src_dpts = torch.stack([i.permute(2, 0, 1) for i in dpt], dim=0)  # for data locality # S, H, W, 1 -> S, 1, H, W
        if norm[0] is not None: output.src_norms = torch.stack([i.permute(2, 0, 1) for i in norm], dim=0)  # for data locality # S, H, W, 3 -> S, 3, H, W

        return output

    def get_ground_truth(self, index):
        # Load actual images, mask, sampling weights
        output = self.get_ibr_metadata(index)

        # Load images
        if not self.skip_loading_images:
            rgb, msk, dpt, norm = self.get_image(output.view_index, output.latent_index)  # H, W, 3
        else:
            rgb, msk, dpt, norm, K, H, W = self.load_image(output.view_index, output.latent_index)  # H, W, 3
            output.K, output.H, output.W = K, H, W
            output.tar_ixt = output.K  # 3, 3

        # Maybe crop images
        if self.immask_crop:  # these variables are only available when loading gts
            meta = dotdict()
            meta.crop_x = self.crop_xs[output.view_index, output.latent_index]
            meta.crop_y = self.crop_ys[output.view_index, output.latent_index]
            meta.orig_h = self.orig_hs[output.view_index, output.latent_index]
            meta.orig_w = self.orig_ws[output.view_index, output.latent_index]
            output.update(meta)
            output.meta.update(meta)

        elif self.imbound_crop:  # crop_x has already been set by imbound_crop for ixts
            x, y, w, h = output.crop_x, output.crop_y, output.W, output.H
            rgb = rgb[y:y + h, x:x + w]
            msk = msk[y:y + h, x:x + w]
            if dpt is not None: dpt = dpt[y:y + h, x:x + w]
            if norm is not None: norm = norm[y:y + h, x:x + w]
            H, W = h, w

        output.rgb = rgb.permute(2, 0, 1)  # (3, H, W), full image in case you need it
        output.msk = msk.permute(2, 0, 1)  # (1, H, W), full mask
        if dpt is not None: output.dpt = dpt.permute(2, 0, 1)  # (1, H, W), full depth map
        if norm is not None: output.norm = norm.permute(2, 0, 1)  # (3, H, W), full normal map

        if dpt is not None:
            # Perform scene scaling, original -> canonical
            dpt, src_dpts = output.dpt * self.scale, output.src_dpts * self.scale
            output.src_exts[:3, 3:] = output.src_exts[:3, 3:] * self.scale
            output.tar_ext[:3, 3:] = output.tar_ext[:3, 3:] * self.scale

            # Perform depth normalization for all the views
            dpt, msk, min, max = normalize_depth(self.depth_norm, dpt, output.n, output.f)
            src_dpts, _, _, _  = normalize_depth(self.depth_norm, src_dpts, output.n, output.f)
            output.dpt, output.src_dpts, output.msk = dpt, src_dpts, output.msk * msk
            output.n, output.f = min, max

        # Crop the image, mask, depth, normal for painless upsampling and skip connection
        H, W = rgb.shape[:2]
        x, y, w, h = get_xywh_from_hwc(H, W, self.latent_factor * self.network_factor)
        output.H, output.W, output.meta.H, output.meta.W = h, w, h, w
        # Target view croppping
        output.tar_ixt, output.rgb, output.msk = crop_using_xywh(x, y, w, h, output.tar_ixt, output.rgb, output.msk)
        if 'dpt' in output: _, output.dpt = crop_using_xywh(x, y, w, h, output.tar_ixt, output.dpt)
        if 'norm' in output: _, output.norm = crop_using_xywh(x, y, w, h, output.tar_ixt, output.norm)
        output.K, output.meta.K = output.tar_ixt, output.tar_ixt  # update K
        # Source views cropping
        output.src_ixts, output.src_inps, output.src_msks = crop_using_xywh(x, y, w, h, output.src_ixts, output.src_inps, output.src_msks)
        if 'src_dpts' in output: _, output.src_dpts = crop_using_xywh(x, y, w, h, output.src_ixts, output.src_dpts)
        if 'src_norms' in output: _, output.src_norms = crop_using_xywh(x, y, w, h, output.src_ixts, output.src_norms)
        output.meta.src_ixts = output.src_ixts  # update src_ixts in meta

        if self.use_epipolar:
            # Compute plucker embeddings and epipolar masks
            epi_msks, plc_embs = compute_masks_pluckers(torch.cat([output.tar_ext[None], output.src_exts], dim=0),
                                                        torch.cat([output.tar_ixt[None], output.src_ixts], dim=0),
                                                        output.H, output.W, self.latent_factor)
            output.epi_mask = epi_msks  # (V * Hz * Wz, V * Hz * Wz)
            output.plc_embs = plc_embs  # (V, 6, Hz, Wz)

        # Other bookkeepings
        im = self.ims[output.view_index, output.latent_index]
        output.meta.rgb_name = im.split('/')[-4] + '_' + im.split('/')[-2] + '_' + os.path.basename(im)  # Image saving name
        output.meta.depth_norm = self.depth_norm.name  # type of depth normalization

        return output

    def __getitem__(self, index: int):  # for now, we are using the dict as list
        # Load ground truth
        output = self.get_ground_truth(index)  # load images, camera parameters, etc (10ms)

        return output

    def __len__(self): return self.sampler_n_views * self.sampler_n_latents  # there's no notion of epoch here

    @staticmethod
    def crop_ixts_bounds(output: dotdict):
        """
        Crops target intrinsics using a xywh computed from a bounds
        """
        x, y, w, h = get_bound_2d_bound(output.bounds, output.K, output.R, output.T, output.meta.H, output.meta.W)
        return MultiViewDataset.crop_ixts(output, x, y, w, h)

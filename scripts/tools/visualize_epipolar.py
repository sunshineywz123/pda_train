import os
import torch
import imageio
import numpy as np
from tqdm import tqdm, trange
import torch.nn.functional as F

import sys
sys.path.append(".")
sys.path.append("..")
from lib.utils.epipolar_utils import compue_mask_plucker

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.easy_utils import read_camera
from easyvolcap.utils.image_utils import resize_image
from easyvolcap.utils.math_utils import affine_inverse
from easyvolcap.utils.data_utils import load_image, load_depth


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default='data/workspace/hypersim_evc/test/ai_001_010/00', help="dataset root directory")
    parser.add_argument("--camera_dir", type=str, default='', help="camera parameters directory")
    parser.add_argument("--images_dir", type=str, default='images', help="where the images should be saved")
    parser.add_argument("--depths_dir", type=str, default='depths', help="where the depths should be saved")
    parser.add_argument("--source_dir", type=str, default='source', help="where the source indices should be saved")
    parser.add_argument("--output_dir", type=str, default="data/workspace/visualize/epipolars", help="the name of the fused point cloud file")

    parser.add_argument("--target_index", type=int, default=0, help="")
    parser.add_argument("--n_srcs", type=int, default=3, help="")
    parser.add_argument("--ratio", type=float, default=0.078125, help="")
    args = parser.parse_args()

    return args


def load_cameras(data_root, intri_file='intri.yml', extri_file='extri.yml'):
    cameras = read_camera(join(data_root, intri_file), join(data_root, extri_file))
    camera_names = np.asarray(sorted(list(cameras.keys())))  # NOTE: sorting camera names
    cameras = dotdict({k: cameras[k] for k in camera_names})

    # NOTE: ALWAYS, ALWAYS, SORT CAMERA NAMES.
    Hs = torch.as_tensor([cameras[k].H for k in camera_names], dtype=torch.float)  # V
    Ws = torch.as_tensor([cameras[k].W for k in camera_names], dtype=torch.float)  # V
    Ks = torch.as_tensor([cameras[k].K for k in camera_names], dtype=torch.float)  # V, 3, 3
    Rs = torch.as_tensor([cameras[k].R for k in camera_names], dtype=torch.float)  # V, 3, 3
    Ts = torch.as_tensor([cameras[k].T for k in camera_names], dtype=torch.float)  # V, 3, 1
    Cs = -Rs.mT @ Ts  # V, 3, 1
    w2cs = torch.cat([Rs, Ts], dim=-1)  # V, 3, 4
    c2ws = affine_inverse(w2cs)  # V, 3, 4

    return camera_names, Hs, Ws, Ks, w2cs, c2ws


def load_sources(data_root, source_dir, target_index, n_srcs=3):
    # Load the source indices
    src_inds = np.load(join(data_root, source_dir, f"src_inds.npy"))  # V, V

    # Get the source indices for the target index
    src_inds = src_inds[target_index, :1+n_srcs]  # V

    return src_inds


def load_images(data_root, images_dir, cameras):
    # Load all the images
    imgs = [load_image(join(data_root, images_dir, cam, '000000.jpg')) for cam in cameras]

    # Concatenate the images
    imgs = np.stack(imgs, axis=0)  # V, H, W, 3

    return imgs


def load_depths(data_root, depths_dir, cameras):
    # Load all the images
    dpts = [load_depth(join(data_root, depths_dir, cam, '000000.exr')) for cam in cameras]

    # Concatenate the images
    dpts = np.stack(dpts, axis=0)  # V, H, W, 1

    return dpts


def main():
    # Parse arguments
    args = parse_args()

    # Load camera parameters
    camera_names, Hs, Ws, Ks, w2cs, c2ws = load_cameras(args.data_root, intri_file='intri.yml', extri_file='extri.yml')

    # Load source indices
    indices = load_sources(args.data_root, args.source_dir, args.target_index, args.n_srcs)

    # Select all the needed cameras
    camera_names = camera_names[indices]
    Hs, Ws, Ks = Hs[indices], Ws[indices], Ks[indices]  # V; V; V, 3, 3
    w2cs, c2ws = w2cs[indices], c2ws[indices]  # V, 3, 4; V, 3, 4

    # Load all the images
    imgs = load_images(args.data_root, args.images_dir, camera_names)  # V, H, W, 3
    # Load all the depths
    dpts = load_depths(args.data_root, args.depths_dir, camera_names)  # V, H, W, 1

    V, Ho, Wo = imgs.shape[0], imgs.shape[1], imgs.shape[2]
    H, W = int(Ho * args.ratio), int(Wo * args.ratio)
    # Resize the images and depths and camera intrinsics
    imgs = resize_image(torch.as_tensor(imgs), size=(H, W)).numpy()  # # V, H, W, 3
    dpts = resize_image(torch.as_tensor(dpts), size=(H, W)).numpy()  # # V, H, W, 1
    Ks[:, :2] = Ks[:, :2] * args.ratio  # V, 3, 3
    # Transpose the images and depths
    imgs = imgs.transpose(0, 3, 1, 2)  # V, 3, H, W
    dpts = dpts.transpose(0, 3, 1, 2)  # V, 1, H, W

    # Create placeholders for the output
    epi_msks = torch.zeros((V, V, H * W, H * W), dtype=torch.bool)  # V, V, H, W

    # Compute the epipolar masks
    for i in trange(V, desc="computing pairwise epipolar masks"):
        for j in range(i + 1, V):
            # Get the source and target images and depths
            tar_img, src_img, tar_dpt, src_dpt = imgs[i], imgs[j], dpts[i], dpts[j]
            tar_ixt, src_ixt, tar_ext, src_ext = Ks[i:i+1], Ks[j:j+1], w2cs[i:i+1], w2cs[j:j+1]
            # Compute the epipolar mask
            tar_msk, src_msk, _, _ = compue_mask_plucker(tar_ext, src_ext, tar_ixt, src_ixt, H, W,
                                                         dialate_mask=True, debug_depth=True, visualize_mask=False)
            # Store the masks
            epi_msks[i, j] = tar_msk
            epi_msks[j, i] = src_msk

    # Visualize epipolar mask
    visualize_epipolar_mask(epi_msks, imgs, V, H, W, args.output_dir)


def visualize_epipolar_mask(epi_msks, imgs, V, H, W, output_dir, step_size=10):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Visualize epipolar mask (print epipolar line and pixel)
    for i in range(V):
        for j in range(V):
            # Skip if same view
            if i == j: continue

            highlights = []
            # Iterate over pixels with step size
            for px in tqdm(range(0, H, step_size), total=(H // step_size), desc="drawing visuals"):
                # `py` is the pixel in the width dimension
                for py in range(0, W, step_size):
                    # Epipolar mask
                    epi_msk = epi_msks[i, j][px * W + py].reshape(H, W)

                    # Highlight pixel in source view
                    tar_img = imgs[i].copy() 
                    tar_img[:, px, py] *= 0
                    # Highlight epipolar mask in target view
                    src_img = imgs[j].copy()
                    src_img[:, epi_msk] *= 0

                    # Concatenate images
                    tar_src_imgs = np.concatenate([tar_img, src_img], axis=2)
                    tar_src_imgs = (tar_src_imgs.transpose(1, 2, 0) * 255).astype(np.uint8)
                    highlights.append(tar_src_imgs)

            # Save gif
            gif_path = os.path.join(output_dir, f"epipolar_src_{i}_tgt_{j}.gif")
            imageio.mimsave(gif_path, highlights, fps=5)


# def main(image_size=196, num_views=3, start_idx=1, object="lynx"):
#     # randomly select an object and load few frames + cameras
#     object_frames_paths = glob.glob(f"data/samples/lynx/images/*.png")
#     object_frames_paths = sorted(object_frames_paths)[start_idx:start_idx+num_views]

#     # load all frames
#     object_frames = [torch.from_numpy(imageio.imread(frame_path)) for frame_path in object_frames_paths]

#     # replace black with white background 
#     object_frames = [torch.where(frame[:,:,-1:] < 255, torch.ones_like(frame) * 255.0, frame) for frame in object_frames]
#     object_frames = [(frame[:,:,:3]).float() / 255.0 for frame in object_frames]

#     # resize to 196 (adjust based on your RAM)
#     size = image_size
#     object_frames = [F.interpolate(frame[None].permute(0,3,1,2), size=(size, size)) for frame in object_frames]
#     object_frames = [frame.squeeze() for frame in object_frames]

#     # load all cameras
#     object_cams_paths = object_frames_paths
#     object_cameras = [np.load(frame_path.replace(".png", ".npy").replace("images", "cameras"), allow_pickle=True) for frame_path in object_cams_paths]
    
#     # field-of-view (fov) and extrinsics (matrix_world)
#     object_cameras_fov = [camera.item()['fov'] for camera in object_cameras]
#     object_cameras = [torch.from_numpy(np.matrix(camera.item()['matrix_world']).reshape(4,4)).float() for camera in object_cameras]

#     # load all depths
#     object_depths = [torch.from_numpy(imageio.imread(frame_path.replace(".png", ".exr").replace("images", "depths"))) for frame_path in object_frames_paths]
#     object_depths = [depth[:, :, :1] * 10.0 for depth in object_depths]
#     object_depths = [F.interpolate(depth[None].permute(0,3,1,2), size=(size, size)).squeeze(0) for depth in object_depths]

#     # placeholders for epipolar attention masks and plucker embeddings
#     num_views = len(object_frames)
#     image_size = max(object_frames[0].shape)
#     epipolar_attention_masks = torch.zeros(num_views, num_views, image_size ** 2, image_size ** 2, dtype=torch.bool)
#     plucker_embeds = [None for _ in range(num_views)]
    
#     # select pairs of source and target frame
#     # compute epipolar attention masks and plucker embeddings b/w each pair
#     for src_idx in trange(len(object_frames), desc="computing pairwise epipolar masks"):
#         for tgt_idx in range(src_idx + 1, len(object_frames)):
#             src_image, tgt_image = object_frames[src_idx], object_frames[tgt_idx]
#             src_camera, tgt_camera = object_cameras[src_idx], object_cameras[tgt_idx]
#             src_depth, tgt_depth = object_depths[src_idx], object_depths[tgt_idx]
#             src_fov, tgt_fov = object_cameras_fov[src_idx], object_cameras_fov[tgt_idx]

#             src_frame = edict({
#                 "camera": src_camera,
#                 "image_rgb": src_image[None], # batch dimension
#                 "depth_map": src_depth[None], # batch dimension
#                 "fov": src_fov
#             })

#             tgt_frame = edict({
#                 "camera": tgt_camera,
#                 "image_rgb": tgt_image[None], # batch dimension
#                 "depth_map": tgt_depth[None], # batch dimension
#                 "fov": tgt_fov
#             })

#             # create attention mask and pluckers, and store them
#             src_mask, tgt_mask, src_plucker, tgt_plucker = get_mask_and_plucker(src_frame, tgt_frame, image_size, dialate_mask=True, debug_depth=True, visualize_mask=False)
            
#             epipolar_attention_masks[src_idx, tgt_idx] = src_mask
#             epipolar_attention_masks[tgt_idx, src_idx] = tgt_mask
#             plucker_embeds[src_idx], plucker_embeds[tgt_idx] = src_plucker, tgt_plucker

#     # visualize epipolar mask
#     visualize_epipolar_mask(epipolar_attention_masks, object_frames, num_views, image_size)


if __name__ == "__main__":
    main()

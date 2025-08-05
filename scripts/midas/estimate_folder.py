import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
sys.path.append('./trdparties/MiDaS')
# sys.path.append('./trdparties/DepthAnything/torchhub')
# sys.path.append('./trdparties/DepthAnything/torchhub/facebookresearch_dinov2_main')
from PIL import Image
import torch
import glob
from lib.utils.vis_utils import colorize_depth_maps
import imageio
from torchvision.transforms import Compose
import cv2
import torch.nn.functional as F
from trdparties.MiDaS.midas.model_loader import default_models, load_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='path to the input folder')
    parser.add_argument('--output', type=str, help='path to output foloder')
    parser.add_argument('--use_disp', action='store_true', help='path to output foloder')
    args = parser.parse_args()
    return args

first_execution = True
def process(device, model, model_type, image, input_size, target_size, optimize, use_camera):
    """
    Run the inference and interpolate.

    Args:
        device (torch.device): the torch device used
        model: the model used for inference
        model_type: the type of the model
        image: the image fed into the neural network
        input_size: the size (width, height) of the neural network input (for OpenVINO)
        target_size: the size (width, height) the neural network output is interpolated to
        optimize: optimize the model to half-floats on CUDA?
        use_camera: is the camera used?

    Returns:
        the prediction
    """
    global first_execution

    if "openvino" in model_type:
        if first_execution or not use_camera:
            print(f"    Input resized to {input_size[0]}x{input_size[1]} before entering the encoder")
            first_execution = False

        sample = [np.reshape(image, (1, 3, *input_size))]
        prediction = model(sample)[model.output(0)][0]
        prediction = cv2.resize(prediction, dsize=target_size,
                                interpolation=cv2.INTER_CUBIC)
    else:
        sample = torch.from_numpy(image).to(device).unsqueeze(0)

        if optimize and device == torch.device("cuda"):
            if first_execution:
                print("  Optimization to half-floats activated. Use with caution, because models like Swin require\n"
                      "  float precision to work properly and may yield non-finite depth values to some extent for\n"
                      "  half-floats.")
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        if first_execution or not use_camera:
            height, width = sample.shape[2:]
            print(f"    Input resized to {width}x{height} before entering the encoder")
            first_execution = False

        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    return prediction


def read_image(path):
    """Read image and output RGB image (0-1).

    Args:
        path (str): path to file

    Returns:
        array: RGB image (0-1)
    """
    img = cv2.imread(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    return img

def predict_depth(img_path, model, transform, device, model_type, net_w, net_h, optimize=False, use_disp=True):
    original_image_rgb = read_image(img_path)
    image = transform({"image": original_image_rgb})["image"]
    with torch.no_grad():
        prediction = process(device, model, model_type, image, (net_w, net_h), original_image_rgb.shape[1::-1], optimize, False)
    if not use_disp:
        prediction = np.clip(prediction, 0.01, None)
        prediction = 1 / prediction
    return prediction

def main(args):
    # repo = "isl-org/ZoeDepth"
    # zoed_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True) 
    # cfg = 'zoedepth_nk' if args.zoe_type == 'nk' else 'zoedepth'
    # conf = get_config(cfg, "infer")
    # zoed_nk = build_model(conf)
    # zoe = zoed_nk.to(DEVICE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = '/home/linhaotong/.cache/torch/hub/checkpoints/dpt_swin2_large_384.pt'
    model_type = 'dpt_swin2_large_384'
    
    
    model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize=False)
    
    imgs_suffix = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
    img_paths = []
    for suffix in imgs_suffix:
        img_paths.extend(glob.glob(join(args.input, '*'+suffix)))
    img_paths = sorted([img_path for img_path in img_paths if img_path[:1] != '.'])
    dpts = []
    for img_path in tqdm(img_paths):
        depth_numpy = predict_depth(img_path, model, transform, device, model_type, net_w, net_h, optimize=False, use_disp=args.use_disp)
        dpts.append(depth_numpy)
    dpts = np.array(dpts)
    dpt_min, dpt_max = dpts.min(), dpts.max()
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(join(args.output, 'npys'), exist_ok=True)
    os.makedirs(join(args.output, 'imgs'), exist_ok=True)
    os.makedirs(join(args.output, 'mp4s'), exist_ok=True)
    
    
    save_imgs = []
    for dpt, img_path in zip(dpts, img_paths):
        npy_path = join(args.output, 'npys/{}'.format(os.path.basename(img_path).split('.')[0]+'.npy'))
        dpt_compressed = np.round(dpt, 5).astype(np.float32)
        np.savez_compressed(npy_path, dpt=dpt_compressed)
        
        dpt_norm = (dpt - dpt_min) / (dpt_max - dpt_min)
        depth_vis = colorize_depth_maps(dpt_norm, 0., 1.)[0].transpose((1, 2, 0))
        save_img = np.concatenate([(depth_vis*255.).astype(np.uint8), np.asarray(Image.open(img_path).convert("RGB"))], axis=1)
        
        img_path = join(args.output, 'imgs/{}'.format(os.path.basename(img_path)))
        imageio.imwrite(img_path, save_img)
        
        save_imgs.append(save_img)
        
    imageio.mimwrite(join(args.output, 'mp4s/output.mp4'), save_imgs, fps=24, quality=7)

if __name__ == '__main__':
    args = parse_args()
    main(args)
import tyro
import os 
from tqdm.auto import tqdm
from os.path import join
import numpy as np
import imageio
from PIL import Image
import torch
from diffusers import AutoencoderKL
import cv2
import torch.nn.functional as F
import sys

from lib.utils.pylogger import Log

def vis_tensor(tensor: torch.Tensor, k: int = 3):
    assert(len(tensor.shape) == 3)
    scaled_data = tensor.reshape(-1, tensor.shape[-1])
    U, S, V = torch.svd(scaled_data)
    principal_components = V[:, :k]
    projected_data = torch.mm(scaled_data, principal_components)
    min_val, max_val = projected_data.min(), projected_data.max()
    projected_data = (projected_data - min_val) / (max_val - min_val)
    return projected_data.reshape(tensor.shape[0], tensor.shape[1], k)

def load_tensor_from_image(img_path: str, resize: bool = False, size: int = 768):
    """
    Args:
        img_path (str): path to image
    Returns:
        img (torch.Tensor): shape (H, W, 3)
    """
    img = Image.open(img_path)
    img_size = img.size
    if resize and max(img_size) != size:
        img = img.resize((int(img_size[0] * size / max(img_size)), int(img_size[1] * size / max(img_size))))
    if img_size[0] % 8 != 0 or img_size[1] % 8 != 0:
        img = np.array(img)[:img_size[1] // 8 * 8, :img_size[0] // 8 * 8]
        print('Image size {} is resized to {}'.format(img_size, img.shape))
    img = torch.from_numpy(np.array(img))
    img = img.permute(2, 0, 1).float() / 255.
    return img[None]

def encode_rgb(rgb_in: torch.Tensor, vae: AutoencoderKL, rgb_latent_scale_factor: float = 0.18215, quantize: bool = True):
    """
    Args:
        rgb_in (torch.Tensor): shape (H, W, 3)
    Returns:
        latent (torch.Tensor): shape (1, 256)
    """
    h = vae.encoder(rgb_in)
    if quantize:
        moments = vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        rgb_latent = mean * rgb_latent_scale_factor
        return rgb_latent
    else:
        return h

@torch.no_grad()
def decode_latent(latent: torch.Tensor, vae: AutoencoderKL, quantize: bool = True, rgb_latent_scale_factor: float = 0.18215):
    if quantize:
        latent = latent / rgb_latent_scale_factor
        z = vae.post_quant_conv(latent)
        return vae.decoder(z)
    return vae.decoder(latent[:, :4])

def rgba2rgb(
        input_path: str = '/mnt/bn/haotongdata/Datasets/nerf_synthetic/lego/test',
        output_path: str = '/mnt/bn/haotongdata/Datasets/nerf_synthetic/lego/test_rgb'
    ) -> None:
    """Parse metric from txt."""
    os.makedirs(output_path, exist_ok=True)
    image_names = sorted(os.listdir(input_path))
    for image_name in tqdm(image_names):
        if image_name.startswith('.DS'): continue
        if 'depth' in image_name: continue
        if 'normal' in image_name: continue
        image = np.asarray(imageio.imread(join(input_path, image_name))) / 255
        rgb, a = image[..., :3], image[..., 3] 
        rgb = rgb * a[..., None] + (1 - a[..., None])
        imageio.imwrite(join(output_path, image_name), (rgb * 255).astype(np.uint8))
    
def rgb2latent(
        input_path: str = '/mnt/bn/haotongdata/Datasets/nerf_llff_data/fern/images_4',
        output_path: str = '/mnt/bn/haotongdata/Datasets/nerf_llff_data/fern/images_4_latent',
        vae_path: str = 'data/pl_htcode/cache_models/vae',
        quantize: bool = True,
        resize: bool = False,
    ) -> None:
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(vae_path)
    vae.eval()
    vae.requires_grad_(False)
    vae.to('cuda')
    
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_path + '_vis', exist_ok=True)
    image_names = sorted(os.listdir(input_path))
    for image_name in tqdm(image_names):
        img_tensor = load_tensor_from_image(join(input_path, image_name), resize=resize).cuda() * 2 - 1.
        latent = encode_rgb(img_tensor, vae, quantize=quantize)
        Log.info('Latent min: {}, max: {}'.format(latent.min(), latent.max()))
        latent_vis = vis_tensor(latent[0].permute(1, 2, 0))
        np.savez_compressed(join(output_path, image_name.replace('.png', '.npz')), data=latent[0].permute((1, 2, 0)).detach().cpu().numpy())
        imageio.imwrite(join(output_path + '_vis', image_name), (latent_vis.detach().cpu().numpy() * 255).astype(np.uint8))
        # plt.imshow(latent_vis.detach().cpu().numpy())
        # plt.savefig('test.jpg')
        # import ipdb; ipdb.set_trace()
        
def latent2rgb(
        input_path: str = '/mnt/bn/haotongdata/home/linhaotong/projects/torch-ngp/trial_fern_sameres_latent_rgb_nogradlatent/results',
        output_path: str = '/mnt/bn/haotongdata/home/linhaotong/projects/torch-ngp/trial_fern_sameres_latent_rgb_nogradlatent/results',
        # input_path: str = '/mnt/bn/haotongdata/Datasets/nerf_llff_data/fern/images_latent',
        # output_path: str = '/mnt/bn/haotongdata/Datasets/nerf_llff_data/fern/images_latent_decode',
        vae_path: str = 'data/pl_htcode/cache_models/vae',
        quantize: bool = True,
        resize: bool = False,
    ) -> None:
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(vae_path)
    vae.eval()
    vae.requires_grad_(False)
    vae.to('cuda')
    
    os.makedirs(output_path, exist_ok=True)
    image_names = sorted(os.listdir(input_path))
    for image_name in tqdm(image_names):
        if '.npz' not in image_name: continue
        if 'latent' not in image_name: continue
        if '443' not in image_name: continue
        latent = np.load(join(input_path, image_name))['data']
        # latent = latent[:189, :252]
        # latent = cv2.resize(latent, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        latent = torch.from_numpy(latent).permute(2, 0, 1).float().cuda()[None]
        # img_tensor = load_tensor_from_image(join(input_path, image_name), resize=resize).cuda() * 2 - 1.
        # latent = encode_rgb(img_tensor, vae, quantize=quantize)
        image = decode_latent(latent, vae, quantize=quantize)
        image = (image * 0.5 + 0.5).permute(0, 2, 3, 1)[0]
        image = image.detach().cpu().numpy()
        image = np.clip(image, 0., 1.)
        image = (image * 255).astype(np.uint8)
        imageio.imwrite(join(output_path, image_name.replace('.npz', '.png').replace('.png', '_leftup.png')), image)
        
def checkvae(
        # input_path: str = '/mnt/bn/haotongdata/Datasets/nerf_llff_data/fern/images_4',
        input_path: str = '/mnt/bn/haotongdata/Datasets/nerf_llff_data/fern/images_4',
        output_path: str = 'test.jpg',
        vae_path: str = 'data/pl_htcode/cache_models/vae',
        quantize: bool = True,
        resize: bool = False,
        crop_ratio: float = 1.,
        code_ratio: float = 1.,
        resize_mode: str = 'bilinear',
    ) -> None:
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(vae_path)
    vae.eval()
    vae.requires_grad_(False)
    vae.to('cuda')
    
    image_names = sorted(os.listdir(input_path))
    for image_name in tqdm(image_names):
        img_tensor = load_tensor_from_image(join(input_path, image_name), resize=resize).cuda() * 2 - 1.
        img = np.asarray(Image.open(join(input_path, image_name)))
        latent = encode_rgb(img_tensor, vae, quantize=quantize)
        if code_ratio != 1.:
            latent = F.interpolate(latent, size=None, scale_factor=code_ratio, mode=resize_mode)
        if crop_ratio != 1.:
            rand_num = np.random.randint(0, (latent.shape[2] - int(latent.shape[2] * crop_ratio)))
            print(rand_num)
            latent = latent[:, :, rand_num:rand_num+int(latent.shape[2] * crop_ratio), rand_num:rand_num + int(latent.shape[3] * crop_ratio)]
        latent_vis = vis_tensor(latent[0].permute(1, 2, 0))
        decode_img = decode_latent(latent, vae, quantize=quantize)
        decode_img = (decode_img * 0.5 + 0.5).permute(0, 2, 3, 1)[0]
        decode_img = decode_img.detach().cpu().numpy()
        decode_img = np.clip(decode_img, 0., 1.)
        
        import matplotlib.pyplot as plt 
        plt.subplot(141)
        plt.imshow(img_tensor[0].permute(1, 2, 0).detach().cpu().numpy() * 0.5 + 0.5)
        plt.axis('off')
        plt.subplot(142)
        plt.imshow(latent_vis.detach().cpu().numpy())
        plt.axis('off')
        plt.subplot(143)
        plt.imshow(decode_img)  
        plt.axis('off')
        if crop_ratio != 1.:
            plt.subplot(144)
            plt.imshow(decode_img[rand_num*8:rand_num*8+int(img.shape[0] * crop_ratio), rand_num*8:rand_num*8+int(img.shape[1] * crop_ratio)])  
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        Log.info('Image saved to {}'.format(output_path))
        sys.exit(0)

if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict(
        {
            "rgba2rgb": rgba2rgb,
            "rgb2latent": rgb2latent,
            "latent2rgb": latent2rgb,
            "checkvae": checkvae,
        }
    )


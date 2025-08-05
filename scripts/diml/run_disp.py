import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')
sys.path.append('trdparties/IGEV/IGEVStereo')
import imageio
import torch
from scripts.warp_disp.utils import warp_image
from trdparties.IGEV.IGEVStereo.core.igev_stereo import IGEVStereo

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained_models/sceneflow/sceneflow.pth')
    parser.add_argument('--dataset', help="dataset for evaluation", default='sceneflow', choices=["eth3d", "kitti", "sceneflow"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--mixed_precision', default=False, action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecure choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")
    args = parser.parse_args()
    return args

def main(args):
    # left_rgb_dir = '/mnt/data/home/linhaotong/research_projects/pl_htcode/data/pl_htcode/DIML/train/LR/outleft'
    # right_rgb_dir = '/mnt/data/home/linhaotong/research_projects/pl_htcode/data/pl_htcode/DIML/train/LR/outright'
    # disp_out_dir = '/mnt/data/home/linhaotong/research_projects/pl_htcode/data/pl_htcode/DIML/train/LR/pred_disp/IGEV'
    # warp_out_dir_depthanything = '/mnt/data/home/linhaotong/research_projects/pl_htcode/data/pl_htcode/DIML/train/LR/warped_img/depth_anything'
    # warp_out_dir_IGEV = '/mnt/data/home/linhaotong/research_projects/pl_htcode/data/pl_htcode/DIML/train/LR/warped_img/IGEV'
    left_rgb_dir = '/mnt/data/home/linhaotong/research_projects/pl_htcode/data/pl_htcode/DIML/test/LR/outleft'
    right_rgb_dir = '/mnt/data/home/linhaotong/research_projects/pl_htcode/data/pl_htcode/DIML/test/LR/outright'
    disp_out_dir = '/mnt/data/home/linhaotong/research_projects/pl_htcode/data/pl_htcode/DIML/test/LR/pred_disp/IGEV'
    warp_out_dir_depthanything = '/mnt/data/home/linhaotong/research_projects/pl_htcode/data/pl_htcode/DIML/test/LR/warped_img/depth_anything'
    warp_out_dir_IGEV = '/mnt/data/home/linhaotong/research_projects/pl_htcode/data/pl_htcode/DIML/test/LR/warped_img/IGEV'
    
    
    os.makedirs(disp_out_dir, exist_ok=True)
    os.makedirs(warp_out_dir_depthanything, exist_ok=True)
    os.makedirs(warp_out_dir_IGEV, exist_ok=True)
    
    left_rgbs = sorted(os.listdir(left_rgb_dir))
    
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    checkpoint = torch.load(args.restore_ckpt)
    model.load_state_dict(checkpoint, strict=True)
    model.cuda()
    model.eval()
    
    
    for img_name in tqdm(left_rgbs):
        left_rgb_path = join(left_rgb_dir, img_name)
        right_rgb_path = join(right_rgb_dir, img_name)
        left_rgb = imageio.imread(left_rgb_path)
        right_rgb = imageio.imread(right_rgb_path)
    
        image1 = torch.from_numpy(np.asarray(left_rgb).transpose(2, 0, 1).astype(np.uint8))
        image2 = torch.from_numpy(np.asarray(right_rgb).transpose(2, 0, 1).astype(np.uint8))
    
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        with torch.no_grad():
            flow_pr = model(image1, image2, iters=32, test_mode=True)
        disp = flow_pr[0, 0].detach().cpu().numpy()
        save_path = join(disp_out_dir, img_name)
        np.savez_compressed(save_path, np.round(disp, 2))
        
        warped_img, msk = warp_image(left_rgb, disp)
        save_path = join(warp_out_dir_IGEV, img_name)
        imageio.imwrite(save_path, warped_img)
        save_path = join(warp_out_dir_IGEV, img_name.split('.')[0] + '_msk.png')
        imageio.imwrite(save_path, (msk * 255.).astype(np.uint8))
        
        warped_img, msk = warp_image((np.asarray(left_rgb)/255.).astype(np.float32), disp_align=disp)
        warped_img = (warped_img*255.).astype(np.uint8)
        save_path = join(warp_out_dir_depthanything, img_name)
        imageio.imwrite(save_path, warped_img)
        save_path = join(warp_out_dir_depthanything, img_name.split('.')[0] + '_msk.png')
        imageio.imwrite(save_path, (msk * 255.).astype(np.uint8))
        
        
    # warped_img2, msk = warp_image((np.asarray(left_rgb)/255.).astype(np.float32), disp_align=disp)
    # imageio.imwrite('warped_img.png', warped_img)
    # imageio.imwrite('warped_img_depthanything.png', (warped_img2*255.).astype(np.uint8))

if __name__ == '__main__':
    args = parse_args()
    main(args)
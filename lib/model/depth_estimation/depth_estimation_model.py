from sklearn import metrics
from lib.model.model import Model
import os
from os.path import join
import numpy as np
import imageio
import torch
import cv2

from lib.utils.base_utils import unproject_depthmap_focal
from lib.utils.dpt.eval_utils import recover_metric_depth, recover_metric_depth_ransac
from lib.utils.parallel_utils import async_call
from lib.utils.pylogger import Log
from lib.utils.vis_utils import colorize_depth_maps
import boxx

class DepthEstimationModel(Model):
    def __init__(
        self,
        pipeline,  # The pipeline is the model itself
        optimizer,  # The optimizer is the optimizer used to train the model
        lr_table,  # The lr_table is the learning rate table
        output_dir: str,
        output_tag: str = 'default',
        clear_output_dir: bool = False,
        scheduler_cfg=None,  # The scheduler_cfg is the scheduler configuration
        ignored_weights_prefix=["pipeline.text_encoder",
                                "pipeline.vae",
                                "pipeline.noise_model"],
        save_orig_pred=False,  # Whether to save the original prediction
        save_vis_depth=False,  # Whether to save the visualized depth
        save_align_pred=False,  # Whether to save the original prediction
        # whether to save the visualized depth and concatenated image
        save_vis_depth_and_concat_img=False,
        save_vis_depth_and_concat_gt=False,
        save_vis_depth_and_concat_err=False,
        save_vis_depth_and_concat_lowres=False,
        save_depth_mesh=False,
        save_align_depth_mesh=False,
        save_gt_depth_mesh=False,
        save_lowres_depth_mesh=False,
        save_lowres_bilinear_depth_mesh=False,
        near_depth=-1., 
        far_depth=-1.,
        compute_abs_metric=False,
        compute_rel_metric=True,
        ransac_align_depth=True,
        focal=1440.,
        concat_axis=1,
        **kwargs,
    ):
        super().__init__(pipeline, optimizer, lr_table, output_dir,
                         output_tag, clear_output_dir, scheduler_cfg, ignored_weights_prefix, **kwargs)

        self._save_orig_pred = save_orig_pred
        self._concat_axis = concat_axis
        self._save_vis_depth = save_vis_depth
        self._save_vis_depth_and_concat_img = save_vis_depth_and_concat_img
        self._save_vis_depth_and_concat_err = save_vis_depth_and_concat_err
        self._save_depth_mesh = save_depth_mesh
        self._save_align_depth_mesh = save_align_depth_mesh
        self._save_gt_depth_mesh = save_gt_depth_mesh
        self._save_lowres_depth_mesh = save_lowres_depth_mesh
        self._save_lowres_bilinear_depth_mesh = save_lowres_bilinear_depth_mesh
        self._compute_abs_metric = compute_abs_metric
        self._compute_rel_metric = compute_rel_metric
        self._save_vis_depth_and_concat_gt = save_vis_depth_and_concat_gt
        self._save_vis_depth_and_concat_lowres = save_vis_depth_and_concat_lowres
        self._focal = focal
        self._near_depth = near_depth
        self._far_depth = far_depth
        self._save_align_pred = save_align_pred
        self.align_depth_func = recover_metric_depth_ransac if ransac_align_depth else recover_metric_depth
        
        Log.info('Results will be saved to: {}'.format(self.output_dir))

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        output = self.pipeline.forward_test(batch)
        if self._save_orig_pred:
            self.save_depth(output['depth'], batch['image_name'], 'orig_pred')
        if self._save_vis_depth:
            lowres_depth = batch['sparse_depth'] if 'sparse_depth' in batch else None
            lowres_depth = batch['lowres_depth'] if 'lowres_depth' in batch and lowres_depth is None else lowres_depth
            self.save_vis_depth(output['depth'], batch['image'], batch['image_name'], 'vis_depth', gt_depth=batch['depth'] if 'depth' in batch else None, lowres_depth=lowres_depth)
            self.save_vis_depth(batch['depth'], batch['image'], batch['image_name'], 'vis_depth_gt', gt_depth=batch['depth'] if 'depth' in batch else None, lowres_depth=batch['lowres_depth'] if 'lowres_depth' in batch else lowres_depth)
        if self._save_depth_mesh and 'disp' not in output:
            self.save_depth_mesh(output['depth'], batch['image'], batch['image_name'], 'pointcloud')
        if self._save_gt_depth_mesh:
            self.save_depth_mesh(batch['depth'], batch['image'], batch['image_name'], 'pointcloud_gt')
        if self._save_lowres_depth_mesh:
            try:
                lowres_depth = output['lowres_depth'] if 'lowres_depth' in output else batch['lowres_depth']
                self.save_depth_mesh(batch['lowres_depth'].detach().cpu().numpy(), batch['image'], batch['image_name'], 'pointcloud_lowres')
            except:
                Log.warn('No lowres depth in the output for lowres save')
        if self._save_lowres_bilinear_depth_mesh:
            try:
                lowres_depth = output['lowres_depth'] if 'lowres_depth' in output else batch['lowres_depth']
                self.save_depth_mesh(batch['lowres_depth'].detach().cpu().numpy(), batch['image'], batch['image_name'], 'pointcloud_lowres_bilinear')
            except:
                Log.warn('No lowres depth in the output for bilinear save')
        return output

    def validation_step(self, batch, batch_idx, dataloader_idx=None) -> None:
        output = self.predict_step(batch, batch_idx, dataloader_idx)
        batch_size = batch['image'].shape[0]
        metrics_dict = self.compute_metrics(output, batch)
        for k, v in metrics_dict.items():
            self.log(f'val/{k}', np.mean(v), on_step=False, on_epoch=True,
                     prog_bar=True if 'l1' in k else False , logger=True, batch_size=batch_size, sync_dist=True)

    def compute_metrics(self, output, batch):
        B = batch['image'].shape[0]
        metrics_dict = {}
        for b in range(B):
            if self._compute_abs_metric or self._compute_rel_metric:
                pred_depth = output['depth'][b][0].detach().cpu().numpy()
                gt_depth = batch['depth'][b][0].detach().cpu().numpy()
                msk = self.create_depth_mask(batch['dataset_name'], gt_depth) 
            
            if self._compute_abs_metric:
                metrics_dict_item = self.compute_depth_metric(pred_depth, gt_depth, msk)
                metrics_dict = self.update_metrics_dict(metrics_dict, metrics_dict_item, 'absolute')
                
            if self._compute_rel_metric:
                pred_depth = self.align_depth_func(pred_depth, gt_depth, msk, disp=('disp' in output and output['disp']))
                if self._save_align_pred:
                    self.save_depth(pred_depth, batch['image_name'], 'align_pred')
                if self._save_align_depth_mesh:
                    self.save_depth_mesh_item(pred_depth, batch['image'][b], batch['image_name'][b], 'pointcloud_align', self._focal)
                metrics_dict_item = self.compute_depth_metric(pred_depth, gt_depth, msk)
                metrics_dict = self.update_metrics_dict(metrics_dict, metrics_dict_item, 'relative')
        return metrics_dict
   
    def update_metrics_dict(self, metrics_dict, metrics_dict_item, prefix):
        for k, v in metrics_dict_item.items():
            if f'{prefix}_{k}' not in metrics_dict:
                metrics_dict[f'{prefix}_{k}'] = []
            metrics_dict[f'{prefix}_{k}'].append(v)
        return metrics_dict
    
    def create_depth_mask(self, dataset_name, gt_depth):
        return gt_depth > 1e-3
    
    def compute_depth_metric(self, pred_depth, gt_depth, msk):
        gt = gt_depth[msk]
        pred = pred_depth[msk]
        thresh = np.maximum((gt / (pred + 1e-5)), (pred / (gt + 1e-5)))
        t11 = (thresh < 1.1).mean()
        d1 = (thresh < 1.25).mean()
        d2 = (thresh < 1.25 ** 2).mean()
        d3 = (thresh < 1.25 ** 3).mean()
        
        l1 = np.mean(np.abs(gt - pred))

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        abs_rel = np.mean(np.abs(gt - pred) / (gt + 1e-5))
        sq_rel = np.mean(((gt - pred)**2) / (gt + 1e-5))

        return {
            't11': t11,
            'd1': d1,
            'd2': d2,
            'd3': d3,
            'abs_rel': abs_rel,
            'sq_rel': sq_rel,
            'rmse': rmse,
            'l1': l1,
        }

    @async_call
    def save_depth(self, depth, name, tag) -> None:
        if not isinstance(depth, torch.Tensor):
            depth = torch.tensor(depth).unsqueeze(0).unsqueeze(0)
        for b in range(len(depth)):
            depth_np = depth[b][0].detach().cpu().numpy()
            save_name = name[b][:-4] + '.npz'
            img_path = join(self.output_dir, f'{tag}/{save_name}')
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            np.savez_compressed(img_path, data=np.round(depth_np, 3))
            
    # @async_call
    def save_depth_mesh(self, depth, rgb, name, tag) -> None:
        for b in range(len(depth)):
            if isinstance(depth, torch.Tensor):
                depth_np = depth[b][0].detach().cpu().numpy()
            else:
                depth_np = depth[b][0]
            self.save_depth_mesh_item(depth_np, rgb[b], name[b], tag, focal=self._focal)
            
    @async_call
    def save_depth_mesh_item(self, depth_np, rgb, save_name, tag, focal) -> None:
        if 'lowres' in tag:
            scale = depth_np.shape[0] / rgb.shape[1]
            focal = focal * scale
        if 'bilinear' in tag and 'lowres' in tag:
            depth_np = cv2.resize(depth_np, (rgb_np.shape[2], rgb_np.shape[1]), interpolation=cv2.INTER_LINEAR)
            focal = focal / scale
        import open3d as o3d
        rgb_np = rgb.detach().cpu().numpy().transpose((1, 2, 0))
        if rgb_np.shape[0] != depth_np.shape[0] or rgb_np.shape[1] != depth_np.shape[1]:
            rgb_np = cv2.resize(rgb_np, (depth_np.shape[1], depth_np.shape[0]), interpolation=cv2.INTER_AREA)
        points = unproject_depthmap_focal(depth_np, focal=focal)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
        pcd.colors = o3d.utility.Vector3dVector(rgb_np.reshape(-1, 3))
        save_path = join(self.output_dir, f'{tag}/{save_name[:-4]}.ply')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        o3d.io.write_point_cloud(save_path, pcd)
        
    @async_call
    def save_vis_depth(self, depth, rgb, name, tag, gt_depth=None, lowres_depth=None) -> None:
        # import ipdb; ipdb.set_trace()
        for b in range(len(depth)):
            depth_np = depth[b][0].detach().cpu().numpy()
            save_name = name[b]
            
            save_imgs = []
            save_img = colorize_depth_maps(depth_np, 
                                           depth_np.min() if self._near_depth < 0. else self._near_depth, 
                                           depth_np.max() if self._far_depth < 0. else self._far_depth
                                           )[0].transpose((1, 2, 0))
            save_imgs.append(save_img)
            if self._save_vis_depth_and_concat_img:
                rgb_np = rgb[b].detach().cpu().numpy().transpose((1, 2, 0))
                rgb_np = cv2.resize(rgb_np, (save_img.shape[1], save_img.shape[0]), interpolation=cv2.INTER_AREA)
                save_img = np.concatenate([rgb_np, save_img], axis=self._concat_axis)
                save_imgs.append(rgb_np)
            if gt_depth is not None and self._save_vis_depth_and_concat_gt:
                gt_depth_np = gt_depth[b][0].detach().cpu().numpy()
                valid_mask = gt_depth_np > 1e-3
                depth_min = gt_depth_np[valid_mask].min()
                depth_max = gt_depth_np[valid_mask].max()
                gt_depth_np = np.clip(gt_depth_np, depth_min, depth_max)
                gt_depth_np_vis = colorize_depth_maps(gt_depth_np, depth_min, depth_max)[0].transpose((1, 2, 0))
                save_img = np.concatenate([save_img, gt_depth_np_vis], axis=self._concat_axis)
                save_imgs.append(gt_depth_np_vis)
                if self._save_vis_depth_and_concat_err:
                    depth_diff = np.abs(depth_np - gt_depth_np)
                    depth_diff[valid_mask] = depth_diff[valid_mask] / gt_depth_np[valid_mask]
                    diff_max, diff_min = depth_diff[valid_mask].max(), depth_diff[valid_mask].min()
                    vis_map = np.zeros_like(depth_diff)
                    vis_map[valid_mask] = depth_diff[valid_mask] 
                    vis_map_vis = colorize_depth_maps(vis_map, diff_min, diff_max)[0].transpose((1, 2, 0))
                    save_img = np.concatenate([save_img, vis_map_vis], axis=self._concat_axis)
                    save_imgs.append(vis_map_vis)
            if lowres_depth is not None and self._save_vis_depth_and_concat_lowres:
                lowres_depth_np = lowres_depth[b][0].detach().cpu().numpy()
                tar_h, tar_w = depth_np.shape[0], depth_np.shape[1]
                if lowres_depth_np.shape[1] != tar_w or lowres_depth_np.shape[0] != tar_h :
                    if (lowres_depth_np == 0.).sum() >= 10:
                        u, v = lowres_depth_np.nonzero()
                        orig_u, orig_v = u, v
                        u, v = (u * tar_h / lowres_depth_np.shape[0]).astype(np.int32), (v * tar_w / lowres_depth_np.shape[1]).astype(np.int32)
                        lowres_depth_np_new = np.zeros_like(depth_np)
                        lowres_depth_np_new[u, v] = lowres_depth_np[orig_u, orig_v]
                        lowres_depth_np = lowres_depth_np_new
                    else:
                        lowres_depth_np = cv2.resize(lowres_depth_np, (tar_w, tar_h), interpolation=cv2.INTER_LINEAR)
                lowres_depth_np = colorize_depth_maps(lowres_depth_np, 
                                                      lowres_depth_np.min() if self._near_depth < 0. else self._near_depth, 
                                                      lowres_depth_np.max() if self._far_depth < 0. else self._far_depth
                                                      )[0].transpose((1, 2, 0))
                lowres_depth_np = cv2.resize(lowres_depth_np, (depth_np.shape[1], depth_np.shape[0]), interpolation=cv2.INTER_LINEAR)
                save_img = np.concatenate([save_img, lowres_depth_np], axis=self._concat_axis)
                save_imgs.append(lowres_depth_np)
            img_path = join(self.output_dir, f'{tag}/{save_name}')
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            imageio.imwrite(img_path.replace('.png', '.jpg'), (save_img * 255.).astype(np.uint8))


from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import hydra

from lib.model.depth_estimation.depth_anything.cspn import CSPN, weights_init
from lib.utils.pylogger import Log
import functools

from .blocks import FeatureFusionBlock, FeatureFusionDepthBlock, _make_scratch
from .loss import L1loss_Gradient_upsample, TrimmedProcrustesLoss

def _make_fusion_block(features, use_bn, size = None, block_type = 'featurefusionblock'):
    blocks = {'featurefusionblock': FeatureFusionBlock, 'featurefusiondepthblock': FeatureFusionDepthBlock}
    return blocks[block_type](
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )
    
    
def get_intermediate_features(x, model, layer_index):
    for i, layer in enumerate(model):
        x = layer(x)
        if i == layer_index:
            return x  # 返回指定层的输出

class DPTHead(nn.Module):
    def __init__(self, nclass, in_channels, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False, block_type='featurefusionblock', output_act='sigmoid'):
        super(DPTHead, self).__init__()
        
        self.nclass = nclass
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn, block_type=block_type)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn, block_type=block_type)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn, block_type=block_type)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn, block_type=block_type)

        head_features_1 = features
        head_features_2 = 32
        
        act_func = nn.Sigmoid() if output_act == 'sigmoid' else nn.Identity()
        
        if nclass > 1:
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_1, nclass, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
            
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
                act_func,
            )
            
    def forward(self, out_features, patch_h, patch_w, depth=None, return_feat=False):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        if depth is not None:
            path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:], depth=depth)
            path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:], depth=depth)
            path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:], depth=depth)
            path_1 = self.scratch.refinenet1(path_2, layer_1_rn, depth=depth)
        else:
            path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
            path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
            path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
            path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out_feat = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out_feat)
        
        if return_feat:
            out_feat = get_intermediate_features(out_feat, self.scratch.output_conv2, 0)
            return out, out_feat
        else:
            return out
        
        
class DepthAnything(nn.Module):
    def __init__(self, encoder='vitl', 
                 features=256, 
                 alpha=0.5, 
                 out_channels=[256, 512, 1024, 1024], 
                 use_bn=False, 
                 use_clstoken=False, 
                 localhub=True,
                 load_pretrain_backbone=None,
                 load_pretrain_net=None,
                 block_type='featurefusionblock',
                 output_act='sigmoid',
                 warp_func=None,
                 add_grad=True,
                 grad_tags=[],
                 cspn=False,
                 disp=False):
        super(DepthAnything, self).__init__()
        
        assert encoder in ['vits', 'vitb', 'vitl', 'vitg']
        self.encoder = encoder
        
        if encoder == 'vitl':
            torch.manual_seed(1)
        else:
            torch.manual_seed(2)
        
        # in case the Internet connection is not stable, please load the DINOv2 locally
        # self.criterion = TrimmedProcrustesLoss(alpha=alpha, trim=0)
        self.criterion = L1loss_Gradient_upsample()
        self.pretrained = torch.hub.load('{}/checkpoints/dinov2'.format(os.getcwd()), 'dinov2_{:}14'.format(encoder), source='local', pretrained=False)
        dim = self.pretrained.blocks[0].attn.qkv.in_features
        self.depth_head = DPTHead(1, dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken, block_type=block_type, output_act=output_act)
        self.warp_func = hydra.utils.instantiate(warp_func) if warp_func is not None else None
        
        self.register_buffer('_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.disp = disp
        self._cspn = cspn
        if self._cspn:
            self.cspn_net = CSPN()
            self.cspn_net.apply(functools.partial(weights_init, mode='trunc'))
        
        if load_pretrain_backbone is not None:
            assert os.path.exists(load_pretrain_backbone)
            Log.info('Load pretrain backbone from {}'.format(load_pretrain_backbone))
            self.pretrained.load_state_dict(torch.load(load_pretrain_backbone))
        
        if load_pretrain_net is not None:
            # import ipdb; ipdb.set_trace()
            # load_pretrain_net = '/mnt/bn/liheyang/model_zoo/metric_v2_vitl_784_hypersim.pth'
            Log.info('Load pretrain network from {}'.format(load_pretrain_net))
            assert os.path.exists(load_pretrain_net)
            model = torch.load(load_pretrain_net, 'cpu')
            if 'model' in model:
                model = model['model']
                model = {k[7:]:model[k] for k in model}# if 'pretrained' in k}
                # model = {k[7:]:model[k] for k in model if 'pretrained' in k}
                strict = False
            elif 'state_dict' in model:
                model = model['state_dict']
                model = {k[9:]:model[k] for k in model}
                strict = False
            else:
                model = model 
                strict = False
            self.load_state_dict(model, strict=strict)
        
        
    def forward(self, x, need_align=False, batch=None):
        h, w = x.shape[-2:]
        
        layer_idxs = {'vitg': [9, 19, 29, 39], 'vitl': [4, 11, 17, 23], 'vitb': [2, 5, 8, 11], 'vits': [2, 5, 8, 11]}
        features = self.pretrained.get_intermediate_layers(x, layer_idxs[self.encoder], return_class_token=True)
        
        patch_h, patch_w = h // 14, w // 14
        
        # depth = self.depth_head(features, patch_h, patch_w)
        depth, depth_feat = self.depth_head(features, patch_h, patch_w, return_feat=True)
        if self._cspn and 'sparse_depth' in batch and (batch['sparse_depth'] != 0).sum() > 10:
            sparse_depth = self.warp_func.warp(batch['sparse_depth'])
            sparse_depth[sparse_depth < 0] = 0.
            if depth.shape[2] != sparse_depth.shape[2]:
                orig_h, orig_w = sparse_depth.shape[2:]
                tar_h, tar_w = depth.shape[2:]
                uv = sparse_depth.nonzero()
                tar_uv = uv.clone()
                tar_uv[:, 2] = (uv[:, 2] * tar_h / orig_h).long()
                tar_uv[:, 3] = (uv[:, 3] * tar_w / orig_w).long()
                tar_sparse_depth = torch.zeros_like(depth)
                tar_sparse_depth[tar_uv[:, 0], tar_uv[:, 1], tar_uv[:, 2], tar_uv[:, 3]] = sparse_depth[uv[:, 0], uv[:, 1], uv[:, 2], uv[:, 3]]
                sparse_depth = tar_sparse_depth
            depth = self.cspn_net(depth_feat, depth, sparse_depth)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)

        if need_align:
            features = features[-1][0].permute(0, 2, 1)
            b, c = features.shape[:2]
            features = features.reshape(b, c, patch_h, patch_w)
            return depth.squeeze(1), features
        
        return depth.squeeze(1)
   
    def compute_gt_disparity(self, batch):
        if 'disparity' in batch: return batch['disparity']
        disparity_gt = batch['depth']
        # disparity_gt[batch['mask'] == 1] = 1 / disparity_gt[batch['mask'] == 1]
        for b in range(len(disparity_gt)):
            msk = batch['mask'][b] == 1
            assert(msk.sum() > 0)
            disparity_gt_min = disparity_gt[b][msk].min()
            disparity_gt_max = disparity_gt[b][msk].max()
            if disparity_gt_max - disparity_gt_min < 1e-6: disparity_gt_max = disparity_gt_min + 1e-6
            disparity_gt[b][msk] = (disparity_gt[b][msk] - disparity_gt_min) / (disparity_gt_max - disparity_gt_min)
        return disparity_gt

    def forward_train_batch(self, batch):
        output = self.forward_test(batch, training=True)
        depth = output['depth']
        depth_gt = self.warp_func.warp(batch['depth'], reference=batch['depth'])
        # loss = self.criterion(1/torch.clip(disparity.squeeze(1), 0.001), disparity_gt.squeeze(1), (batch['mask']==1).squeeze(1).float())
        loss = self.criterion(depth, depth_gt, (batch['mask']==1).float(), add_grad=False)[0]
        return {'loss': loss}
    
    def forward_train(self, batches):
        if not isinstance(batches, List):
            return self.forward_train_batch(batches)
        loss = 0.
        outputs = []
        for batch in batches:
            outputs.append(self.forward_train_batch(batch))
            loss += outputs[-1]["loss"]
        return {'loss': loss}
        
    def forward_test(self, batch, training=False):
        depth = self.forward((batch['image'] - self._mean) / self._std, batch=batch)
        if self.disp:
            return {'depth': depth[None], 'disp': True}
        else:
            if 'depth' in batch and depth.shape[2:] != batch['depth'].shape[2:]:
                depth = F.interpolate(depth.unsqueeze(1), size=batch['depth'].shape[2:], mode='bilinear', align_corners=False)
            if not training: depth = self.warp_func.unwarp(depth, reference=batch['depth']) 
            return {'depth': depth}
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from hydra.utils import instantiate

from lib.utils.pylogger import Log

from .blocks import FeatureFusionBlock, FeatureFusionDepthBlock, _make_scratch
from .loss import TrimmedProcrustesLoss

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


class DPTHead(nn.Module):
    def __init__(self, nclass, in_channels, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False, block_type='featurefusionblock',
                 use_sigmoid=False):
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
        
        if nclass > 1:
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_1, nclass, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
            if use_sigmoid: 
                self.scratch.output_conv2 = nn.Sequential(
                    nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(True),
                    nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
                    nn.Sigmoid(),
                )
            else:
                self.scratch.output_conv2 = nn.Sequential(
                    nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(True),
                    nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(True),
                    nn.Identity(),
                )
            
    def forward(self, out_features, patch_h, patch_w, depth=None):
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
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        
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
                 loss_cfg=None,
                 use_sigmoid=False):
        super(DepthAnything, self).__init__()
        
        assert encoder in ['vits', 'vitb', 'vitl', 'vitg']
        self.encoder = encoder
        
        if encoder == 'vitl':
            torch.manual_seed(1)
        else:
            torch.manual_seed(2)
        
        # in case the Internet connection is not stable, please load the DINOv2 locally
        if loss_cfg is None:
            self.criterion = TrimmedProcrustesLoss(alpha=alpha, trim=0)
        else:
            self.criterion = instantiate(loss_cfg)
        self.pretrained = torch.hub.load('{}/cache_models/depth_anything/torchhub/facebookresearch_dinov2_main'.format(os.environ['workspace']), 'dinov2_{:}14'.format(encoder), source='local', pretrained=False)
        dim = self.pretrained.blocks[0].attn.qkv.in_features
        self.depth_head = DPTHead(1, dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken, block_type=block_type, use_sigmoid=use_sigmoid)
        
        if load_pretrain_backbone is not None:
            assert os.path.exists(load_pretrain_backbone)
            Log.info('Load pretrain backbone from {}'.format(load_pretrain_backbone))
            model = torch.load(load_pretrain_backbone, 'cpu')
            if 'model' in model: model = {k[18:]:model['model'][k] for k in model['model'] if 'pretrained' in k}
            self.pretrained.load_state_dict(model)
        
        if load_pretrain_net is not None:
            # import ipdb; ipdb.set_trace()
            # load_pretrain_net = '/mnt/bn/liheyang/model_zoo/metric_v2_vitl_784_hypersim.pth'
            Log.info('Load pretrain network from {}'.format(load_pretrain_net))
            assert os.path.exists(load_pretrain_net)
            model = torch.load(load_pretrain_net, 'cpu')['model']
            model = {k[7:]:model[k] for k in model } # if 'pretrained' in k}
            self.load_state_dict(model, strict=False)
        
        self.register_buffer('_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self._use_sigmoid = use_sigmoid
        
    def forward(self, x, need_align=False):
        h, w = x.shape[-2:]
        
        layer_idxs = {'vitg': [9, 19, 29, 39], 'vitl': [4, 11, 17, 23], 'vitb': [2, 5, 8, 11], 'vits': [2, 5, 8, 11]}
        features = self.pretrained.get_intermediate_layers(x, layer_idxs[self.encoder], return_class_token=True)
        
        patch_h, patch_w = h // 14, w // 14

        depth = self.depth_head(features, patch_h, patch_w)
        if self._use_sigmoid:
            depth = depth * 20
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)

        if need_align:
            features = features[-1][0].permute(0, 2, 1)
            b, c = features.shape[:2]
            features = features.reshape(b, c, patch_h, patch_w)
            return depth.squeeze(1), features
        
        return depth.squeeze(1)

    def forward_train_batch(self, batch):
        depth = self.forward((batch['image'] - self._mean) / self._std).unsqueeze(1)
        loss = self.criterion(depth.squeeze(1), batch['depth'].squeeze(1), (batch['mask']==1).squeeze(1).float())
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
        
    def forward_test(self, batch):
        import ipdb; ipdb.set_trace()
        depth = self.forward((batch['image'] - self._mean) / self._std)
        depth = F.interpolate(depth.unsqueeze(1), size=batch['depth'].shape[2:], mode='bilinear', align_corners=False)
        return {'depth': depth}
from typing import List
from torch import nn
import torch
import torch.nn.functional as F
import os

from lib.utils.pylogger import Log

from .MultiScaleDepthSR import MultiscaleDepthDecoder
from .densenet import DenseNet121

def div_by_mask_sum(loss: torch.Tensor, mask_sum: torch.Tensor):
    return loss / torch.max(mask_sum, torch.ones_like(mask_sum))


class SafeTorchLog(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        input_abs = torch.abs(input) + 1e-9
        ctx.save_for_backward(input_abs)

        return torch.log(input_abs)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        (input_abs,) = ctx.saved_tensors
        grad_input = grad_output.clone()

        return grad_input * (1.0 / input_abs) / 2.302585093  # ln(10)


safe_torch_log = SafeTorchLog.apply


def create_gradient_log_loss(log_prediction_d, mask, log_gt):

    # compute log difference
    log_d_diff = log_prediction_d - log_gt
    log_d_diff = torch.mul(log_d_diff, mask)

    # compute vertical gradient
    v_gradient = torch.abs(log_d_diff[:, :, 2:, :] - log_d_diff[:, :, :-2, :])
    v_mask = torch.mul(mask[:, :, 2:, :], mask[:, :, :-2, :])
    v_gradient = torch.mul(v_gradient, v_mask)

    # compute horizontal gradient
    h_gradient = torch.abs(log_d_diff[:, :, :, 2:] - log_d_diff[:, :, :, :-2])
    h_mask = torch.mul(mask[:, :, :, 2:], mask[:, :, :, :-2])
    h_gradient = torch.mul(h_gradient, h_mask)

    # sum up gradients
    grad_loss = torch.sum(h_gradient, dim=[1, 2, 3]) + torch.sum(v_gradient, dim=[1, 2, 3])
    num_valid_pixels = torch.sum(mask, dim=[1, 2, 3])
    grad_loss = div_by_mask_sum(grad_loss, num_valid_pixels)

    return grad_loss


def create_gradient_log_loss_4_scales(log_prediction, log_ground_truth, mask):
    log_prediction_d = log_prediction
    log_gt = log_ground_truth
    mask = mask

    log_prediction_d_scale_1 = log_prediction_d[:, :, ::2, ::2]
    log_prediction_d_scale_2 = log_prediction_d_scale_1[:, :, ::2, ::2]
    log_prediction_d_scale_3 = log_prediction_d_scale_2[:, :, ::2, ::2]

    mask_scale_1 = mask[:, :, ::2, ::2]
    mask_scale_2 = mask_scale_1[:, :, ::2, ::2]
    mask_scale_3 = mask_scale_2[:, :, ::2, ::2]

    log_gt_scale_1 = log_gt[:, :, ::2, ::2]
    log_gt_scale_2 = log_gt_scale_1[:, :, ::2, ::2]
    log_gt_scale_3 = log_gt_scale_2[:, :, ::2, ::2]

    gradient_loss_scale_0 = create_gradient_log_loss(log_prediction_d, mask, log_gt)

    gradient_loss_scale_1 = create_gradient_log_loss(
        log_prediction_d_scale_1, mask_scale_1, log_gt_scale_1
    )

    gradient_loss_scale_2 = create_gradient_log_loss(
        log_prediction_d_scale_2, mask_scale_2, log_gt_scale_2
    )

    gradient_loss_scale_3 = create_gradient_log_loss(
        log_prediction_d_scale_3, mask_scale_3, log_gt_scale_3
    )

    gradient_loss_4_scales = (
        gradient_loss_scale_0 + gradient_loss_scale_1 + gradient_loss_scale_2 + gradient_loss_scale_3
    )

    return gradient_loss_4_scales

class MSPF(nn.Module):
    """
    Inspired by: Multi-Scale Progressive Fusion Learning for Depth Map Super-Resolution
    https://arxiv.org/pdf/2011.11865v1.pdf

    variables:
        decoder_channel_output_scales: available scale factors for reducing channels
        in decoder based on encoder input channels.
    params:
        decoder_channel_scale: used to control decoder size

    """

    decoder_channel_output_scales = [1, 2, 4, 8, 16]

    def __init__(self, upsample_factor=8, decoder_channel_scale=2, pretrain_path=None):
        super(MSPF, self).__init__()
        self.encoder = DenseNet121()
        assert decoder_channel_scale in self.decoder_channel_output_scales, \
            f"decoder scale factor not supported {decoder_channel_scale} - supported {self.decoder_channel_output_scales}"
        input_channels = self.encoder.skip_out_channels[::-1]
        output_channels = [int(ch/decoder_channel_scale) for ch in self.encoder.skip_out_channels[::-1]]

        self.decoder = MultiscaleDepthDecoder(input_channels, output_channels, upsample_factor)
        
        self._upsample_factor = upsample_factor
        
        if pretrain_path is not None and os.path.exists(pretrain_path):
            Log.info('Load pretrained model: {}'.format(pretrain_path))
            self.load_state_dict(torch.load(pretrain_path, 'cpu')['model'])

    def forward(self, batch):
        rgb = batch['image'] - 0.5
        depth = batch['lowres_depth']
        skip_features = self.encoder(rgb)
        output = self.decoder(depth, skip_features)
        return output
    
    def forward_train_batch(self, batch):
        prediction = self.forward(batch)
        valid_mask = batch['mask'] 
        gt_depth = batch['depth']
        
        
        error_image = torch.abs(prediction - gt_depth) * valid_mask.float()
        sum_loss = torch.sum(error_image, dim=[1, 2, 3])
        num_valid_pixels = torch.sum(valid_mask, dim=[1, 2, 3])
        l1_loss = sum_loss / torch.max(num_valid_pixels, torch.ones_like(num_valid_pixels) * 1e-6)
        l1_loss = torch.mean(l1_loss)
        
        
        log_prediction = safe_torch_log(prediction)
        log_gt = safe_torch_log(gt_depth)
        grad_loss = create_gradient_log_loss_4_scales(log_prediction, log_gt, valid_mask)
        grad_loss = torch.mean(grad_loss)
        
        loss = l1_loss + 2*grad_loss
        
        return {'loss': loss, 'l1_loss_vis': l1_loss, 'grad_loss_vis': grad_loss}
    
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
        # lowres_depth_ = batch['lowres_depth'].clone()
        h, w = batch['image'].shape[2:]
        if h//self._upsample_factor != batch['lowres_depth'].shape[2]:
            batch['lowres_depth'] = F.interpolate(batch['lowres_depth'], size=(h//self._upsample_factor, w//self._upsample_factor), mode='bilinear', align_corners=False)
        depth = self.forward(batch)
        # batch['lowres_depth'] = lowres_depth_
        return {'depth': depth}

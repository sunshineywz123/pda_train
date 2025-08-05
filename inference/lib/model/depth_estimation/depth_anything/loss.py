import torch
from torch import nn
import torch.nn.functional as F
EPS = 1e-3

class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        valid_mask = ((target > EPS) & (pred > EPS) & (valid_mask>0.5)).detach() 
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))
        return loss
    

class L1Loss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        l1_diff = F.l1_loss(pred, target, reduction='none')
        l1_diff_sum = (l1_diff * valid_mask.float()).sum(dim=(-1, -2)) 
        l1_sum = valid_mask.float().sum(dim=(-1, -2))
        loss = (l1_diff_sum / l1_sum).mean()
        return loss
    
class L1loss_Gradient(nn.Module):
    def __init__(self, gradient_weight=2., scales=4, reduction='batch-based', normalize=False):
        super().__init__()
        self._gradient_weight = gradient_weight
        self.l1_loss = L1Loss()
        self._normalize = normalize
        if self._gradient_weight > 0.:
            self.gradient_loss = GradientLoss(scales=scales, reduction=reduction)
        
    def forward(self, pred, target, valid_mask):
        if self._normalize:
            target, _, _ = normalize_prediction_minmax(target, valid_mask)
            pred = pred / 20.
        l1_loss = self.l1_loss(pred, target, valid_mask)
        gradient_loss = self.gradient_loss(pred, target, valid_mask)
        return l1_loss + 2. * gradient_loss
    
class L1loss_Gradient_upsample(nn.Module):
    def __init__(self, gradient_weight=2., scales=4, reduction='batch-based'):
        super().__init__()
        self._gradient_weight = gradient_weight
        self.l1_loss = L1Loss()
        self.gradient_loss = GradientLoss(scales=scales, reduction=reduction)
        
    def forward(self, pred, target, valid_mask, add_grad=False, **kwargs):
        l1_loss = self.l1_loss(pred, target, valid_mask)
        loss = l1_loss
        loss_dict = {'l1_loss': l1_loss}
        if add_grad:
            gradient_loss = self.gradient_loss(pred, target, valid_mask)
            loss_dict.update({'grad_loss': gradient_loss})
            if kwargs.get('grad_weight', None ) is not None:
                loss += kwargs.get('grad_weight', None ) * gradient_loss
            else:
                loss += self._gradient_weight * gradient_loss
        return loss, loss_dict
    
    
class L1Logloss_Gradient_upsample(nn.Module):
    def __init__(self, gradient_weight=2., scales=4, reduction='batch-based'):
        super().__init__()
        self._gradient_weight = gradient_weight
        self.l1_loss = L1Loss()
        self.gradient_loss = GradientLoss(scales=scales, reduction=reduction)
        self.log_loss = SiLogLoss()
        
    def forward(self, pred, target, valid_mask, add_grad=False, **kwargs):
        if target.shape[2] == 364: 
            log_loss = self.log_loss(pred, target, valid_mask)
            loss = log_loss
            loss_dict = {'log_loss': log_loss}
        else: 
            l1_loss = self.l1_loss(pred, target, valid_mask)
            loss = l1_loss
            loss_dict = {'l1_loss': l1_loss}
        if add_grad:
            gradient_loss = self.gradient_loss(pred, target, valid_mask)
            loss_dict.update({'grad_loss': gradient_loss})
            if kwargs.get('grad_weight', None ) is not None:
                loss += kwargs.get('grad_weight', None ) * gradient_loss
            else:
                loss += self._gradient_weight * gradient_loss
        return loss, loss_dict

class L1loss_Gradient_meshdepth_zip(nn.Module):
    def __init__(self, gradient_weight=2., scales=4, reduction='batch-based'):
        super().__init__()
        self._gradient_weight = gradient_weight
        self.l1_loss = L1Loss()
        self.gradient_loss = GradientLoss(scales=scales, reduction=reduction)
        
    def forward(self, pred, target, valid_mask, add_grad=False, mesh_depth=None, mesh_mask=None, **kwargs):
        if mesh_depth is not None: 
            if mesh_mask.to(torch.uint8).sum() == 0: l1_loss = 0.
            else: l1_loss = self.l1_loss(pred, mesh_depth, mesh_mask.to(torch.uint8))
        else: l1_loss = self.l1_loss(pred, target, valid_mask)
        loss = l1_loss
        loss_dict = {'l1_loss': l1_loss}
        if add_grad:
            if mesh_mask is not None and mesh_mask.sum() > 100: valid_mask = mesh_mask
            gradient_loss = self.gradient_loss(pred, target, valid_mask)
            loss_dict.update({'grad_loss': gradient_loss})
            if kwargs.get('grad_weight', None ) is not None:
                loss += kwargs.get('grad_weight', None ) * gradient_loss
            else:
                loss += self._gradient_weight * gradient_loss
            # if mesh_depth is not None: loss += 0.1 * gradient_loss
            # else: loss += self._gradient_weight * gradient_loss
        return loss, loss_dict
    
class L1loss_Gradient_meshdepth_zip_seg(nn.Module):
    def __init__(self, gradient_weight=2., scales=4, reduction='batch-based'):
        super().__init__()
        self._gradient_weight = gradient_weight
        self.l1_loss = L1Loss()
        self.gradient_loss = GradientLoss(scales=scales, reduction=reduction)
        
    def forward(self, pred, target, valid_mask, add_grad=False, mesh_depth=None, mesh_mask=None, seg=None):
        if mesh_depth is not None:
            interest_classes =  {'wall': 0, 'sky': 2,  'floor': 3, 'ceiling': 5, 'windowpane': 8, 'door': 14, 'mirror': 27, 'screen': 130, 'crt_screen': 141, 'glass': 147}
            interest_mask = torch.ones_like(seg)
            for k, v in interest_classes.items():
                interest_mask[seg == v] = 0
            new_target = target.clone()
            new_mask = valid_mask.clone()
            new_target[interest_mask.bool()] = mesh_depth[interest_mask.bool()]
            new_mask[interest_mask.bool()] = mesh_mask[interest_mask.bool()].to(torch.uint8)
            if new_mask.sum() == 0: l1_loss = 0.
            else: l1_loss = self.l1_loss(pred, new_target, new_mask)
        else:
            l1_loss = self.l1_loss(pred, target, valid_mask)
        loss = l1_loss
        loss_dict = {'l1_loss': l1_loss}
        if add_grad:
            gradient_loss = self.gradient_loss(pred, target, valid_mask)
            loss_dict.update({'grad_loss': gradient_loss})
            loss += self._gradient_weight * gradient_loss
        return loss, loss_dict
    
class Minmax_Gradient_L1loss(nn.Module):
    def __init__(self, gradient_weight=2., scales=4, l1_weight=0.1, trim=0., reduction='batch-based'):
        super().__init__()
        self._gradient_weight = gradient_weight
        self._l1_weight = l1_weight
        self.l1_loss = L1Loss()
        self.gradient_loss = GradientLoss(scales=scales, reduction=reduction)
        self.trimmed_mae_loss = TrimmedMAELoss(trim=trim)
        
    def forward(self, pred, target, valid_mask):
        normalized_target, val_min, val_max = normalize_prediction_minmax(target, valid_mask)
        normalized_pred, _, _ = normalize_prediction_minmax(pred, valid_mask, val_min, val_max)
        data_loss = self.trimmed_mae_loss(normalized_pred, normalized_target, valid_mask)
        gradient_loss = self.gradient_loss(normalized_pred, normalized_target, valid_mask)
        l1_loss = self.l1_loss(normalized_pred, normalized_target, valid_mask)
        return data_loss + self._gradient_weight * gradient_loss + self._l1_weight * l1_loss

def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def trimmed_mae_loss(prediction, target, mask, trim=0.2, reduction=reduction_batch_based):
    M = torch.sum(mask, (1, 2))
    res = prediction - target

    res = res[mask.bool()].abs()

    trimmed, _ = torch.sort(res.view(-1), descending=False)[
        : int(len(res) * (1.0 - trim))
    ]
    return reduction(trimmed, M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


def normalize_prediction_minmax(target, mask, val_min=None, val_max=None):
    if val_min is None:
        B = len(target)
        val_min, val_max = [], []
        for b in range(B):
            min_ = target[b][mask[b] > 0.5].min()
            max_ = target[b][mask[b] > 0.5].max()
            if (max_ - min_) < EPS: max_ = min_ + EPS
            val_min.append(min_)
            val_max.append(max_)
        val_min = torch.tensor(val_min).to(target.device)[:, None, None]
        val_max = torch.tensor(val_max).to(target.device)[:, None, None]
    return (target - val_min) / (val_max - val_min), val_min, val_max

def normalize_prediction_robust(target, mask):
    ssum = torch.sum(mask, (1, 2))
    valid = ssum > 0

    m = torch.zeros_like(ssum)
    s = torch.ones_like(ssum)

    m[valid] = torch.median(
        (mask[valid] * target[valid]).view(valid.sum(), -1), dim=1
    ).values
    target = target - m.view(-1, 1, 1)

    sq = torch.sum(mask * target.abs(), (1, 2))
    s[valid] = torch.clamp((sq[valid] / ssum[valid]), min=1e-6)

    return target / (s.view(-1, 1, 1))


class TrimmedProcrustesLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, trim=0.2, reduction="batch-based"):
        super(TrimmedProcrustesLoss, self).__init__()

        self.__data_loss = TrimmedMAELoss(reduction=reduction, trim=trim)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask, return_item=False):
        self.__prediction_ssi = normalize_prediction_robust(prediction, mask)
        target_ = normalize_prediction_robust(target, mask)

        total = 0.
        depth_loss = self.__data_loss(self.__prediction_ssi, target_, mask)
        total += depth_loss
        if self.__alpha > 0:
            reg_loss = self.__alpha * self.__regularization_loss(
                self.__prediction_ssi, target_, mask
            )
            total += reg_loss
        if return_item:
            return total, depth_loss, reg_loss
        else:
            return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)


class TrimmedMAELoss(nn.Module):
    def __init__(self, trim=0.2, reduction='batch-based'):
        super().__init__()

        self.trim = trim
        
        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based
            
    def forward(self, prediction, target, mask):
        return trimmed_mae_loss(prediction, target, mask, trim=self.trim, reduction=self.__reduction)
        

class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total
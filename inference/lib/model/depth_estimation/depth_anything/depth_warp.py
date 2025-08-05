import torch
import torch.nn as nn
import math
EPS = 1e-3

class WarpIdentity:
    def warp(self, depth, **kwargs):
        return depth
    def unwarp(self, depth, **kwargs):
        return depth
    
    
class WarpFixMinMax:
    def __init__(self, 
                 near_depth = 1.,
                 far_depth = 80.,
                 **kwargs):
        self._near_depth = near_depth
        self._far_depth = far_depth
        
    def warp(self, depth, **kwargs):
        return (depth - self._near_depth) / (self._far_depth - self._near_depth)
    
    def unwarp(self, depth, **kwargs):
        return depth * (self._far_depth - self._near_depth) + self._near_depth
    

class WarpMinMax:
    def warp(self, depth, reference, **kwargs):
        depth_min, depth_max = reference.reshape(depth.shape[0], -1).min(1, keepdim=True)[0], reference.reshape(depth.shape[0], -1).max(1, keepdim=True)[0]
        if ((depth_max - depth_min) < EPS).any(): 
            depth_max[(depth_max - depth_min)<EPS] = depth_min[(depth_max - depth_min)<EPS] + EPS
        return (depth - depth_min[:, None, None]) / (depth_max - depth_min)[:, None, None]
    def unwarp(self, depth, reference, **kwargs):
        depth_min, depth_max = reference.reshape(depth.shape[0], -1).min(1, keepdim=True)[0], reference.reshape(depth.shape[0], -1).max(1, keepdim=True)[0]
        if ((depth_max - depth_min) < EPS).any(): 
            depth_max[(depth_max - depth_min)<EPS] = depth_min[(depth_max - depth_min)<EPS] + EPS
        return depth * (depth_max - depth_min)[:, None, None] + depth_min[:, None, None]
    
class WarpLogFix:
    def __init__(self, 
                 near_depth = 1.,
                 far_depth = 80.,
                 **kwargs):
        self._near_depth = near_depth
        self._far_depth = far_depth
    def warp(self, depth, **kwargs):
        depth = torch.clamp(depth, self._near_depth, self._far_depth)
        return torch.log(depth / self._near_depth) / math.log(self._far_depth / self._near_depth)
    def unwarp(self, depth, **kwargs):
        return torch.exp(depth * math.log(self._far_depth / self._near_depth)) * self._near_depth
    
class WarpLog:
    def warp(self, depth, reference, **kwargs):
        depth_min, depth_max = reference.reshape(depth.shape[0], -1).min(1, keepdim=True)[0], reference.reshape(depth.shape[0], -1).max(1, keepdim=True)[0]
        if ((depth_max - depth_min) < EPS).any(): 
            depth_max[(depth_max - depth_min)<EPS] = depth_min[(depth_max - depth_min)<EPS] + EPS
        depth = torch.clamp(depth, depth_min[:, None, None], depth_max[:, None, None])
        return torch.log((depth / depth_min[:, None, None]) / torch.log(depth_max / depth_min)[:, None, None])
    
    def unwarp(self, depth, reference, **kwargs):
        depth_min, depth_max = reference.reshape(depth.shape[0], -1).min(1, keepdim=True)[0], reference.reshape(depth.shape[0], -1).max(1, keepdim=True)[0]
        if ((depth_max - depth_min) < EPS).any(): 
            depth_max[(depth_max - depth_min)<EPS] = depth_min[(depth_max - depth_min)<EPS] + EPS
        return torch.exp(depth * torch.log(depth_max / depth_min)[:, None, None]) * depth_min[:, None, None]
    
        
    
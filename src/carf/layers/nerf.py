import torch
import numpy as np
import os, sys, time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import tqdm
from easydict import EasyDict as edict
import cv2
from src.carf.utils import util, camera


class NeRF(torch.nn.Module):
    def __init__(self, cfg):
        super(NeRF, self).__init__()
        input_3D_dim = 3 + 6 * cfg.architecture.pos_encoding.L_3D if cfg.architecture.pos_encoding else 3
        if cfg.nerf.view_dep:
            input_view_dim = 3 + 6 * cfg.architecture.pos_encoding.L_view if cfg.architecture.pos_encoding else 3

        # point-wise feature
        self.mlp_feat = torch.nn.ModuleList()
        L = util.get_layer_dims(cfg.architecture.layers_feat)
        for li, (k_in, k_out) in enumerate(L):
            if li == 0: k_in = input_3D_dim
            if li in cfg.architecture.skip: k_in += input_3D_dim
            if li == len(L) - 1: k_out += 1
            linear = torch.nn.Linear(k_in, k_out)
            if cfg.architecture.tf_init:
                self.tensorflow_init_weights(linear, out="first" if li == len(L) - 1 else None)
            self.mlp_feat.append(linear)

        # RGB prediction
        self.mlp_rgb = torch.nn.ModuleList()
        L = util.get_layer_dims(cfg.architecture.layers_rgb)
        feat_dim = cfg.architecture.layers_feat[-1]
        for li, (k_in, k_out) in enumerate(L):
            if li == 0: k_in = feat_dim + (input_view_dim if cfg.nerf.view_dep else 0) + 3
            linear = torch.nn.Linear(k_in, k_out)
            if cfg.architecture.tf_init:
                self.tensorflow_init_weights(linear, out="all" if li == len(L) - 1 else None)
            self.mlp_rgb.append(linear)

        if cfg.c2f is not None:
            self.progress = torch.nn.Parameter(torch.tensor(0.))  # use Parameter so it could be checkpointed

    @staticmethod
    def tensorflow_init_weights(linear, out=None):
        # use Xavier init instead of Kaiming init
        relu_gain = torch.nn.init.calculate_gain("relu")  # sqrt(2)
        if out == "all":
            torch.nn.init.xavier_uniform_(linear.weight)
        elif out == "first":
            torch.nn.init.xavier_uniform_(linear.weight[:1])
            torch.nn.init.xavier_uniform_(linear.weight[1:], gain=relu_gain)
        else:
            torch.nn.init.xavier_uniform_(linear.weight, gain=relu_gain)
        torch.nn.init.zeros_(linear.bias)

    def forward(self, cfg, points_3D, ray_unit=None, mode=None):  # [B,...,3]
        if cfg.architecture.pos_encoding:
            points_enc = self.positional_encoding(cfg, points_3D, L=cfg.architecture.pos_encoding.L_3D, c2f=True)
            points_enc = torch.cat([points_3D, points_enc], dim=-1)  # [B,...,6L+3]
        else:
            points_enc = points_3D
        feat = points_enc
        # extract coordinate-based features
        for li, layer in enumerate(self.mlp_feat):
            if li in cfg.architecture.skip:
                feat = torch.cat([feat, points_enc], dim=-1)
            feat = layer(feat)
            if li == len(self.mlp_feat) - 1:
                density = feat[..., 0]
                density_activ = getattr(torch_F, cfg.architecture.density_activ)  # relu_,abs_,sigmoid_,exp_....
                density = density_activ(density)
                feat = feat[..., 1:]
            feat = torch_F.relu(feat)

        # predict RGB values
        if cfg.nerf.view_dep:
            assert (ray_unit is not None)
            if cfg.architecture.pos_encoding:
                ray_enc = self.positional_encoding(cfg, ray_unit, L=cfg.architecture.pos_encoding.L_view)
                ray_enc = torch.cat([ray_unit, ray_enc], dim=-1)  # [B,...,6L+3]
            else:
                ray_enc = ray_unit
            feat = torch.cat([feat, ray_enc, points_3D], dim=-1)
        else:
            feat = torch.cat([feat, points_3D], dim=-1)

        for li, layer in enumerate(self.mlp_rgb):
            feat = layer(feat)
            if li != len(self.mlp_rgb) - 1:
                feat = torch_F.relu(feat)
        rgb = feat.sigmoid_()  # [B,...,3]
        return rgb, density

    def forward_samples(self, cfg, center, ray, depth_samples, mode=None):
        # This points are measured in world frame, depth_samples from difference view have no effect
        points_3D_samples = camera.get_3D_points_from_depth(cfg,
                                                            center,
                                                            ray,
                                                            depth_samples,
                                                            multi_samples=True)  # [B,HW,N,3]
        if cfg.nerf.view_dep:
            ray_unit = torch_F.normalize(ray, dim=-1)  # [B,HW,3]
            ray_unit_samples = ray_unit[..., None, :].expand_as(points_3D_samples)  # [B,HW,N,3]
        else:
            ray_unit_samples = None
        rgb_samples, density_samples = self.forward(cfg, points_3D_samples, ray_unit=ray_unit_samples,
                                                    mode=mode)  # [B,HW,N],[B,HW,N,3]
        return rgb_samples, density_samples

    @staticmethod
    def composite(cfg, ray, rgb_samples, density_samples, depth_samples):
        ray_length = ray.norm(dim=-1, keepdim=True)  # [B,HW,1]
        # volume rendering: compute probability (using quadrature)
        depth_intv_samples = depth_samples[..., 1:, 0] - depth_samples[..., :-1, 0]  # [B,HW,N-1]
        depth_intv_samples = torch.cat([depth_intv_samples, torch.empty_like(depth_intv_samples[..., :1]).fill_(1e10)],
                                       dim=2)  # [B,HW,N]
        dist_samples = depth_intv_samples * ray_length  # [B,HW,N]
        sigma_delta = density_samples * dist_samples  # [B,HW,N]
        alpha = 1 - (-sigma_delta).exp_()  # [B,HW,N]
        T = (-torch.cat([torch.zeros_like(sigma_delta[..., :1]), sigma_delta[..., :-1]], dim=2).cumsum(
            dim=2)).exp_()  # [B,HW,N], cumulative summation along the ray direction, which is dim 2 in this scenario
        prob = (T * alpha)[..., None]  # [B,HW,N,1]
        # integrate RGB and depth weighted by probability
        depth = (depth_samples * prob).sum(dim=2)  # [B,HW,1]
        rgb = (rgb_samples * prob).sum(dim=2)  # [B,HW,3]
        opacity = prob.sum(dim=2)  # [B,HW,1]
        return rgb, depth, opacity, prob  # [B,HW,K]

    def positional_encoding(self, cfg, x, L, c2f=False):  # [B,...,N]
        shape = x.shape
        freq = 2 ** torch.arange(L, dtype=torch.float32, device=cfg.device) * np.pi  # [L]
        spectrum = x[..., None] * freq  # [B,...,N,L]
        sin, cos = spectrum.sin(), spectrum.cos()  # [B,...,N,L]
        input_enc = torch.stack([sin, cos], dim=-2)  # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1], -1)  # [B,...,2NL]
        if cfg.c2f is not None and c2f:
            # set weights for different frequency bands
            start, end = cfg.c2f
            alpha = (self.progress.data - start) / (end - start) * L
            k = torch.arange(L, dtype=torch.float32, device=cfg.device)
            weight = (1 - (alpha - k).clamp_(min=0, max=1).mul_(np.pi).cos_()) / 2
            # apply weights
            shape = input_enc.shape
            input_enc = (input_enc.view(-1, L) * weight).view(*shape)
        return input_enc
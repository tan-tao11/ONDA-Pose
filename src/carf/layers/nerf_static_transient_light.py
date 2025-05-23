# MIT License

# Copyright (c) 2023 Hanzhi Chen

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from ..utils import util
import numpy as np
import os, sys, time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import tqdm
from easydict import EasyDict as edict
import cv2
from ..utils import camera


class NeRF(torch.nn.Module):
    def __init__(self, cfg):
        super(NeRF, self).__init__()
        input_3D_dim = 3 + 6 * cfg.architecture.posenc.L_3D if cfg.architecture.posenc else 3
        if cfg.nerf.view_dep:
            input_view_dim = 3 + 6 * cfg.architecture.posenc.L_view if cfg.architecture.posenc.L_view else 3

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
        # Freeze parameters for static scene
        self._freeze_module(self.mlp_feat)

        # RGB prediction
        self.mlp_rgb = torch.nn.ModuleList()
        L = util.get_layer_dims(cfg.architecture.layers_rgb)
        feat_dim = cfg.architecture.layers_feat[-1]
        input_3D_dim_rgb = 3  # + 6 * cfg.architecture.posenc.L_3D_rgb if cfg.architecture.posenc else 3

        for li, (k_in, k_out) in enumerate(L):
            if li == 0: k_in = feat_dim + (input_view_dim if cfg.nerf.view_dep else 0) \
                               + input_3D_dim_rgb + cfg.nerf.N_latent_light

            linear = torch.nn.Linear(k_in, k_out)
            if cfg.architecture.tf_init:
                self.tensorflow_init_weights(linear, out="all" if li == len(L) - 1 else None)
            self.mlp_rgb.append(linear)

        # Transient prediction
        if cfg.architecture.layers_trans:
            self.mlp_trans = torch.nn.ModuleList()
            L = util.get_layer_dims(cfg.architecture.layers_trans)
            for li, (k_in, k_out) in enumerate(L):
                if li == 0: k_in = feat_dim + cfg.nerf.N_latent_trans
                linear = torch.nn.Linear(k_in, k_out)
                if cfg.architecture.tf_init:
                    self.tensorflow_init_weights(linear, out="all" if li == len(L) - 1 else None)
                self.mlp_trans.append(linear)
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

    def forward(self, cfg, points_3D, ray_unit=None, latent_variable_trans=None, latent_variable_light=None,
                mode=None):  # [B,...,3]
        B, HW, N, _ = points_3D.shape

        if cfg.architecture.posenc:
            points_enc = self.positional_encoding(cfg, points_3D, L=cfg.architecture.posenc.L_3D, c2f=True)
            points_enc = torch.cat([points_3D, points_enc], dim=-1)  # [B,...,6L+3]
        else:
            points_enc = points_3D
        feat = points_enc

        with torch.no_grad():
            # extract coordinate-based features
            for li, layer in enumerate(self.mlp_feat):
                if li in cfg.architecture.skip:
                    feat = torch.cat([feat, points_enc], dim=-1)
                feat = layer(feat)
                if li == len(self.mlp_feat) - 1:
                    density = feat[..., 0]
                    if cfg.nerf.density_noise_reg and mode == "train":
                        density += torch.randn_like(density) * cfg.nerf.density_noise_reg
                    density_activ = getattr(torch_F, cfg.architecture.density_activ)  # relu_,abs_,sigmoid_,exp_....
                    density = density_activ(density)
                    feat = feat[..., 1:]
                feat = torch_F.relu(feat)

        # predict RGB values
        if cfg.nerf.view_dep:
            assert (ray_unit is not None)
            if cfg.architecture.posenc.L_view:
                ray_enc = self.positional_encoding(cfg, ray_unit, L=cfg.architecture.posenc.L_view, c2f=True)
                ray_enc = torch.cat([ray_unit, ray_enc], dim=-1)  # [B,...,6L+3]
            else:
                ray_enc = ray_unit
            feat_rgb = torch.cat([feat, ray_enc, points_3D], dim=-1)
        else:
            feat_rgb = torch.cat([feat, points_3D], dim=-1)
        if cfg.nerf.N_latent_light > 0:
            latent_variable_light = latent_variable_light[:, None, None, :].expand(B, HW, N,
                                                                                cfg.nerf.N_latent_light)  # B x HW X N x 32
            feat_rgb = torch.cat([feat_rgb, latent_variable_light], dim=-1)
        else:
            feat_rgb = feat_rgb
        for li, layer in enumerate(self.mlp_rgb):
            feat_rgb = layer(feat_rgb)
            if li != len(self.mlp_rgb) - 1:
                feat_rgb = torch_F.relu(feat_rgb)
        rgb = feat_rgb.sigmoid_()  # [B,...,3]

        # predict uncertainty
        if cfg.architecture.layers_trans:
            latent_variable_trans = latent_variable_trans[:, None, None, :].expand(B, HW, N,
                                                                                   cfg.nerf.N_latent_trans)  # B x HW X N x 32
            feat_trans = torch.cat([feat, latent_variable_trans], dim=-1)  # B x HW X N x (256+32)

            for li, layer in enumerate(self.mlp_trans):
                feat_trans = layer(feat_trans)
                if li != len(self.mlp_trans) - 1:
                    feat_trans = torch_F.relu(feat_trans)

            rgb_trans = (feat_trans[..., :3]).sigmoid_()
            density_trans = torch_F.softplus(feat_trans[..., 3])
            uncert = torch_F.softplus(feat_trans[..., -1].unsqueeze(-1))

            # Stack them to save
            rgb = torch.stack([rgb, rgb_trans], dim=-1)  # B x HW x N x 3 x 2
            density = torch.stack([density, density_trans], dim=-1)  # B x HW x N x 2
        else:
            uncert = None

        return rgb, density, uncert

    def forward_samples(self, cfg, center, ray, depth_samples,
                        latent_variable_trans=None, latent_variable_light=None, mode=None):
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
        rgb_samples, density_samples, uncert_samples = self.forward(cfg,
                                                                    points_3D_samples,
                                                                    ray_unit=ray_unit_samples,
                                                                    latent_variable_trans=latent_variable_trans,
                                                                    latent_variable_light=latent_variable_light,
                                                                    mode=mode)  # [B,HW,N],[B,HW,N,3]
        return rgb_samples, density_samples, uncert_samples

    @staticmethod
    def composite(cfg, ray, rgb_samples, density_samples, depth_samples, uncert_samples=None):
        ray_length = ray.norm(dim=-1, keepdim=True)  # [B,HW,1]
        # volume rendering: compute probability (using quadrature)
        depth_intv_samples = depth_samples[..., 1:, 0] - depth_samples[..., :-1, 0]  # [B,HW,N-1]
        depth_intv_samples = torch.cat([depth_intv_samples, torch.empty_like(depth_intv_samples[..., :1]).fill_(1e10)],
                                       dim=2)  # [B,HW,N]
        dist_samples = depth_intv_samples * ray_length  # [B,HW,N]

        # if len(rgb_samples.shape) == 5:
        sigma_delta_static = density_samples[..., 0] * dist_samples  # [B,HW,N]
        sigma_delta_transient = density_samples[..., -1] * dist_samples  # [B,HW,N]
        sigma_delta = sigma_delta_static + sigma_delta_transient

        alpha_static = 1 - (-sigma_delta_static).exp_()
        alpha_transient = 1 - (-sigma_delta_transient).exp_()
        alpha = 1 - (-sigma_delta).exp_()

        T = (-torch.cat([torch.zeros_like(sigma_delta[..., :1]),
                         sigma_delta[..., :-1]], dim=2).cumsum(dim=2)).exp_()  # [B,HW,N]
        T_static = (-torch.cat([torch.zeros_like(sigma_delta_static[..., :1]),
                                sigma_delta_static[..., :-1]], dim=2).cumsum(dim=2)).exp_()  # [B,HW,N]
        T_transient = (-torch.cat([torch.zeros_like(sigma_delta_transient[..., :1]),
                                   sigma_delta_transient[..., :-1]], dim=2).cumsum(dim=2)).exp_()  # [B,HW,N]

        prob_static = (T * alpha_static)[..., None]  # [B,HW,N,1]
        prob_transient = (T * alpha_transient)[..., None]  # [B,HW,N,1]
        prob = (T * alpha)[..., None]  # [B,HW,N,1]

        # opacity prediction
        opacity = prob.sum(dim=2)  # [B,HW,1]
        opacity_static = (T_static * alpha_static)[..., None].sum(dim=2)
        opacity_transient = (T_transient * alpha_transient)[..., None].sum(dim=2)

        # rgb prediction
        rgb = (rgb_samples[..., 0] * prob_static + rgb_samples[..., -1] * prob_transient).sum(dim=2)  # [B,HW,3]
        rgb_static = ((T_static * alpha_static)[..., None] * rgb_samples[..., 0]).sum(dim=2)  # [B,HW,3]
        rgb_transient = ((T_transient * alpha_transient)[..., None] * rgb_samples[..., -1]).sum(dim=2)  # [B,HW,3]

        uncert = (uncert_samples * prob_transient).sum(dim=2) + cfg.nerf.min_uncert  # [B,HW,1]
        depth = (depth_samples * (T_static * alpha_static)[..., None]).sum(dim=2)  # [B,HW,1]

        return (rgb, rgb_static, rgb_transient, depth, opacity, opacity_static, opacity_transient,
                prob, uncert, alpha_static, alpha_transient)
        # else:
        #     return (rgb, rgb_static, rgb_transient, depth, opacity_static, prob, uncert, alpha_static, alpha_transient)


    def positional_encoding(self, cfg, x, L, c2f=False):  # [B,...,N]
        shape = x.shape
        freq = 2 ** torch.arange(L, dtype=torch.float32, device=cfg.device) * np.pi  # [L]
        spectrum = x[..., None] * freq  # [B,...,N,L]
        sin, cos = spectrum.sin(), spectrum.cos()  # [B,...,N,L]
        input_enc = torch.stack([sin, cos], dim=-2)  # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1], -1)  # [B,...,2NL]
        if cfg.c2f.range is not None and c2f:
            # set weights for different frequency bands
            satrt_freq = 0 if cfg.c2f.start is None else cfg.c2f.start
            start, end = cfg.c2f.range
            alpha = (self.progress.data - start) / (end - start) * L
            k = torch.arange(L, dtype=torch.float32, device=cfg.device) - satrt_freq
            weight = (1 - (alpha - k).clamp_(min=0, max=1).mul_(np.pi).cos_()) / 2
            # apply weights
            shape = input_enc.shape
            input_enc = (input_enc.view(-1, L) * weight).view(*shape)
        return input_enc

    @staticmethod
    def _freeze_module(module):
        for name, param in module.named_parameters():
            param.requires_grad = False

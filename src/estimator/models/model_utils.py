import logging
import copy
import torch
import numpy as np
from ..utils.pose_utils import re, te
from ..utils.pose_utils import rot6d_to_mat_batch


logger = logging.getLogger(__name__)


def get_xyz_mask_out_dim(cfg):
    net_cfg = cfg.MODEL.POSE_NET
    g_head_cfg = net_cfg.GEO_HEAD
    loss_cfg = net_cfg.LOSS_CFG

    xyz_loss_type = loss_cfg.XYZ_LOSS_TYPE
    mask_loss_type = loss_cfg.MASK_LOSS_TYPE
    if xyz_loss_type in ["MSE", "L1", "L2", "SmoothL1"]:
        r_out_dim = 3
    elif xyz_loss_type in ["CE_coor", "CE"]:
        r_out_dim = 3 * (g_head_cfg.XYZ_BIN + 1)
    else:
        raise NotImplementedError(f"unknown xyz loss type: {xyz_loss_type}")

    if mask_loss_type in ["L1", "BCE", "RW_BCE", "dice"]:
        mask_out_dim = 1
    elif mask_loss_type in ["CE"]:
        mask_out_dim = 2
    else:
        raise NotImplementedError(f"unknown mask loss type: {mask_loss_type}")

    return r_out_dim, mask_out_dim

def get_rot_mat(rot, rot_type):
    
    if rot_type in ["ego_rot6d", "allo_rot6d"]:
        rot_m = rot6d_to_mat_batch(rot)
    else:
        raise ValueError(f"Wrong pred_rot type: {rot_type}")
    return rot_m


def get_mask_prob(pred_mask, mask_loss_type):
    # (b,c,h,w)
    # output: (b, 1, h, w)
    bs, c, h, w = pred_mask.shape
    if mask_loss_type == "L1":
        assert c == 1, c
        mask_max = torch.max(pred_mask.view(bs, -1), dim=-1)[0].view(bs, 1, 1, 1)
        mask_min = torch.min(pred_mask.view(bs, -1), dim=-1)[0].view(bs, 1, 1, 1)
        # [0, 1]
        mask_prob = (pred_mask - mask_min) / (mask_max - mask_min)  # + 1e-5)
    elif mask_loss_type in ["BCE", "RW_BCE", "dice"]:
        assert c == 1, c
        mask_prob = torch.sigmoid(pred_mask)
    elif mask_loss_type == "CE":
        mask_prob = torch.softmax(pred_mask, dim=1, keepdim=True)[:, 1:2, :, :]
    else:
        raise NotImplementedError(f"Unknown mask loss type: {mask_loss_type}")
    return mask_prob


def compute_mean_re_te(pred_transes, pred_rots, gt_transes, gt_rots):
    pred_transes = pred_transes.detach().cpu().numpy()
    pred_rots = pred_rots.detach().cpu().numpy()
    gt_transes = gt_transes.detach().cpu().numpy()
    gt_rots = gt_rots.detach().cpu().numpy()

    bs = pred_rots.shape[0]
    R_errs = np.zeros((bs,), dtype=np.float32)
    T_errs = np.zeros((bs,), dtype=np.float32)
    for i in range(bs):
        R_errs[i] = re(pred_rots[i], gt_rots[i])
        T_errs[i] = te(pred_transes[i], gt_transes[i])
    return R_errs.mean(), T_errs.mean()

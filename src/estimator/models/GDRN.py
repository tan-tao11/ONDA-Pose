import copy
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.solver_utils import build_optimizer_with_params
from detectron2.utils.events import get_event_storage
from mmcv.runner import load_checkpoint

from ..losses.coor_cross_entropy import CrossEntropyHeatmapLoss
from ..losses.l2_loss import L2Loss
from ..losses.mask_losses import weighted_ex_loss_probs, soft_dice_loss
from ..losses.pm_loss import PyPMLoss
from ..losses.rot_loss import angular_distance, rot_l2_loss
from .model_utils import compute_mean_re_te, get_mask_prob
from .pose_from_pred_centroid_z import pose_from_pred_centroid_z
from ..utils.common import get_xyz_mask_region_out_dim

from .model_utils import (
    compute_mean_re_te,
    get_mask_prob,
    get_rot_mat,
)

logger = logging.getLogger(__name__)


class GDRN(nn.Module):
    def __init__(self, config, backbone, geo_head_net, neck=None, pnp_net=None):
        super().__init__()
        assert config.model.pose_net.name == "GDRN", config.model.pose_net.name
        self.backbone = backbone
        self.neck = neck

        self.geo_head_net = geo_head_net
        self.pnp_net = pnp_net

        self.config = config
        self.xyz_out_dim, self.mask_out_dim, self.region_out_dim = get_xyz_mask_region_out_dim(config)

        # uncertainty multi-task loss weighting
        # https://github.com/Hui-Li/multi-task-learning-example-PyTorch/blob/master/multi-task-learning-example-PyTorch.ipynb
        # a = log(sigma^2)
        # L*exp(-a) + a  or  L*exp(-a) + log(1+exp(a))
        # self.log_vars = nn.Parameter(torch.tensor([0, 0], requires_grad=True, dtype=torch.float32).cuda())
        # yapf: disable
        if config.model.pose_net.use_mtl:
            self.loss_names = [
                "mask", "coor_x", "coor_y", "coor_z", "coor_x_bin", "coor_y_bin", "coor_z_bin", "region",
                "PM_R", "PM_xy", "PM_z", "PM_xy_noP", "PM_z_noP", "PM_T", "PM_T_noP",
                "centroid", "z", "trans_xy", "trans_z", "trans_LPnP", "rot", "bind",
            ]
            for loss_name in self.loss_names:
                self.register_parameter(
                    f"log_var_{loss_name}", nn.Parameter(torch.tensor([0.0], requires_grad=True, dtype=torch.float32))
                )
        # yapf: enable

    def train(self, mode=True):
        """Override the default train() to freeze the BN parameters."""
        super(GDRN, self).train(mode)
        config = self.config
        if config.model.freeze_bn:
            logger.info("freeze bn...")
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()  # do not update running stats

                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
        return self

    def forward(
        self,
        x,
        gt_xyz=None,
        gt_xyz_bin=None,
        gt_mask_trunc=None,
        gt_mask_visib=None,
        gt_mask_obj=None,
        gt_mask_full=None,
        gt_region=None,
        gt_ego_rot=None,
        gt_points=None,
        sym_infos=None,
        gt_trans=None,
        gt_trans_ratio=None,
        roi_classes=None,
        roi_coord_2d=None,
        roi_coord_2d_rel=None,
        roi_cams=None,
        roi_centers=None,
        roi_whs=None,
        roi_extents=None,
        resize_ratios=None,
        do_loss=False,
        do_self=False,
    ):
        """
        Args:
            do_self: forward for self-supervised training
        """
        config = self.config
        net_config = config.model.pose_net
        g_head_config = net_config.geo_head
        pnp_net_config = net_config.pnp_net

        device = x.device
        bs = x.shape[0]
        num_classes = net_config.num_classes
        out_res = net_config.output_res

        # x.shape [bs, 3, 256, 256]
        conv_feat = self.backbone(x)  # [bs, c, 8, 8]
        if self.neck is not None:
            conv_feat = self.neck(conv_feat)
        mask, coor_x, coor_y, coor_z, region = self.geo_head_net(conv_feat)

        if g_head_config.xyz_class_aware:
            assert roi_classes is not None
            coor_x = coor_x.view(bs, num_classes, self.xyz_out_dim // 3, out_res, out_res)
            coor_x = coor_x[torch.arange(bs).to(device), roi_classes]
            coor_y = coor_y.view(bs, num_classes, self.xyz_out_dim // 3, out_res, out_res)
            coor_y = coor_y[torch.arange(bs).to(device), roi_classes]
            coor_z = coor_z.view(bs, num_classes, self.xyz_out_dim // 3, out_res, out_res)
            coor_z = coor_z[torch.arange(bs).to(device), roi_classes]

        if g_head_config.mask_class_aware:
            assert roi_classes is not None
            mask = mask.view(bs, num_classes, self.mask_out_dim, out_res, out_res)
            mask = mask[torch.arange(bs).to(device), roi_classes]

        if g_head_config.region_class_aware:
            assert roi_classes is not None
            region = region.view(bs, num_classes, self.region_out_dim, out_res, out_res)
            region = region[torch.arange(bs).to(device), roi_classes]

        # -----------------------------------------------
        # get rot and trans from pnp_net
        # NOTE: use softmax for bins (the last dim is bg)
        if coor_x.shape[1] > 1 and coor_y.shape[1] > 1 and coor_z.shape[1] > 1:
            coor_x_softmax = F.softmax(coor_x[:, :-1, :, :], dim=1)
            coor_y_softmax = F.softmax(coor_y[:, :-1, :, :], dim=1)
            coor_z_softmax = F.softmax(coor_z[:, :-1, :, :], dim=1)
            coor_feat = torch.cat([coor_x_softmax, coor_y_softmax, coor_z_softmax], dim=1)
        else:
            coor_feat = torch.cat([coor_x, coor_y, coor_z], dim=1)  # BCHW

        if pnp_net_config.with_2d_coord:
            if pnp_net_config.coord_2d_type == "rel":
                assert roi_coord_2d_rel is not None
                coor_feat = torch.cat([coor_feat, roi_coord_2d_rel], dim=1)
            else:  # default abs
                assert roi_coord_2d is not None
                coor_feat = torch.cat([coor_feat, roi_coord_2d], dim=1)

        # NOTE: for region, the 1st dim is bg
        region_softmax = F.softmax(region[:, 1:, :, :], dim=1)

        mask_atten = None
        if pnp_net_config.mask_attention != "none":
            mask_atten = get_mask_prob(mask, mask_loss_type=net_config.loss_config.mask_loss_type)

        region_atten = None
        if pnp_net_config.region_attention:
            region_atten = region_softmax

        pred_rot_, pred_t_ = self.pnp_net(
            coor_feat, region=region_atten, extents=roi_extents, mask_attention=mask_atten
        )

        # convert pred_rot to rot mat -------------------------
        rot_type = pnp_net_config.rot_type
        pred_rot_m = get_rot_mat(pred_rot_, rot_type)

        # convert pred_rot_m and pred_t to ego pose -----------------------------
        if pnp_net_config.trans_type == "centroid_z":
            pred_ego_rot, pred_trans = pose_from_pred_centroid_z(
                pred_rot_m,
                pred_centroids=pred_t_[:, :2],
                pred_z_vals=pred_t_[:, 2:3],  # must be [B, 1]
                roi_cams=roi_cams,
                roi_centers=roi_centers,
                resize_ratios=resize_ratios,
                roi_whs=roi_whs,
                eps=1e-4,
                is_allo="allo" in rot_type,
                z_type=pnp_net_config.z_type,
                # NOTE: get differentiable pose for training and self-supervised training
                is_train=(do_loss or do_self),
            )
        else:
            raise ValueError(f"Unknown trans type: {pnp_net_config.trans_type}")

        # forward for self-supervised training # -----------------------------------------------
        if do_self:
            out_dict = {"rot": pred_ego_rot, "trans": pred_trans}
            # NOTE: mask is raw mask
            # "region": region
            if mask_atten is not None:
                mask_prob = mask_atten
            else:
                mask_prob = get_mask_prob(mask, mask_loss_type=config.train.loss.mask_loss_type)
            out_dict.update(
                {
                    "mask": mask,
                    "mask_prob": mask_prob,
                    "coor_x": coor_x,
                    "coor_y": coor_y,
                    "coor_z": coor_z,
                    "region": region,
                }
            )
            return out_dict
        # --------------------------------------------------------------------------------------
        if not do_loss:  # test
            out_dict = {"rot": pred_ego_rot, "trans": pred_trans}
            if config.test.use_pnp or config.test.save_results_only:
                # TODO: move the pnp/ransac inside forward
                out_dict.update({"mask": mask, "coor_x": coor_x, "coor_y": coor_y, "coor_z": coor_z, "region": region})
        else:
            out_dict = {}
            assert (
                (gt_xyz is not None)
                and (gt_trans is not None)
                and (gt_trans_ratio is not None)
                and (gt_region is not None)
            )
            mean_re, mean_te = compute_mean_re_te(pred_trans, pred_rot_m, gt_trans, gt_ego_rot)
            vis_dict = {
                "vis/error_R": mean_re,
                "vis/error_t": mean_te * 100,  # cm
                "vis/error_tx": np.abs(pred_trans[0, 0].detach().item() - gt_trans[0, 0].detach().item()) * 100,  # cm
                "vis/error_ty": np.abs(pred_trans[0, 1].detach().item() - gt_trans[0, 1].detach().item()) * 100,  # cm
                "vis/error_tz": np.abs(pred_trans[0, 2].detach().item() - gt_trans[0, 2].detach().item()) * 100,  # cm
                "vis/tx_pred": pred_trans[0, 0].detach().item(),
                "vis/ty_pred": pred_trans[0, 1].detach().item(),
                "vis/tz_pred": pred_trans[0, 2].detach().item(),
                "vis/tx_net": pred_t_[0, 0].detach().item(),
                "vis/ty_net": pred_t_[0, 1].detach().item(),
                "vis/tz_net": pred_t_[0, 2].detach().item(),
                "vis/tx_gt": gt_trans[0, 0].detach().item(),
                "vis/ty_gt": gt_trans[0, 1].detach().item(),
                "vis/tz_gt": gt_trans[0, 2].detach().item(),
                "vis/tx_rel_gt": gt_trans_ratio[0, 0].detach().item(),
                "vis/ty_rel_gt": gt_trans_ratio[0, 1].detach().item(),
                "vis/tz_rel_gt": gt_trans_ratio[0, 2].detach().item(),
            }

            loss_dict = self.gdrn_loss(
                config=self.config,
                out_mask=mask,
                gt_mask_trunc=gt_mask_trunc,
                gt_mask_visib=gt_mask_visib,
                gt_mask_obj=gt_mask_obj,
                out_x=coor_x,
                out_y=coor_y,
                out_z=coor_z,
                gt_xyz=gt_xyz,
                gt_xyz_bin=gt_xyz_bin,
                out_region=region,
                gt_region=gt_region,
                out_trans=pred_trans,
                gt_trans=gt_trans,
                out_rot=pred_ego_rot,
                gt_rot=gt_ego_rot,
                out_centroid=pred_t_[:, :2],  # TODO: get these from trans head
                out_trans_z=pred_t_[:, 2],
                gt_trans_ratio=gt_trans_ratio,
                gt_points=gt_points,
                sym_infos=sym_infos,
                extents=roi_extents,
                # roi_classes=roi_classes,
            )

            if net_config.use_mtl:
                for _name in self.loss_names:
                    if f"loss_{_name}" in loss_dict:
                        vis_dict[f"vis_lw/{_name}"] = torch.exp(-getattr(self, f"log_var_{_name}")).detach().item()
            for _k, _v in vis_dict.items():
                if "vis/" in _k or "vis_lw/" in _k:
                    if isinstance(_v, torch.Tensor):
                        _v = _v.item()
                    vis_dict[_k] = _v
            storage = get_event_storage()
            storage.put_scalars(**vis_dict)

            return out_dict, loss_dict
        return out_dict

    def gdrn_loss(
        self,
        config,
        out_mask,
        gt_mask_trunc,
        gt_mask_visib,
        gt_mask_obj,
        out_x,
        out_y,
        out_z,
        gt_xyz,
        gt_xyz_bin,
        out_region,
        gt_region,
        out_rot=None,
        gt_rot=None,
        out_trans=None,
        gt_trans=None,
        out_centroid=None,
        out_trans_z=None,
        gt_trans_ratio=None,
        gt_points=None,
        sym_infos=None,
        extents=None,
    ):
        net_config = config.model.pose_net
        g_head_config = net_config.geo_head
        pnp_net_config = net_config.pnp_net
        loss_config = config.train.loss_config

        loss_dict = {}

        gt_masks = {"trunc": gt_mask_trunc, "visib": gt_mask_visib, "obj": gt_mask_obj}

        # xyz loss ----------------------------------
        if not g_head_config.freeze:
            xyz_loss_type = loss_config.xyz_loss_type
            gt_mask_xyz = gt_masks[loss_config.xyz_loss_mask_gt]
            if xyz_loss_type == "L1":
                loss_func = nn.L1Loss(reduction="sum")
                loss_dict["loss_coor_x"] = loss_func(
                    out_x * gt_mask_xyz[:, None], gt_xyz[:, 0:1] * gt_mask_xyz[:, None]
                ) / gt_mask_xyz.sum().float().clamp(min=1.0)
                loss_dict["loss_coor_y"] = loss_func(
                    out_y * gt_mask_xyz[:, None], gt_xyz[:, 1:2] * gt_mask_xyz[:, None]
                ) / gt_mask_xyz.sum().float().clamp(min=1.0)
                loss_dict["loss_coor_z"] = loss_func(
                    out_z * gt_mask_xyz[:, None], gt_xyz[:, 2:3] * gt_mask_xyz[:, None]
                ) / gt_mask_xyz.sum().float().clamp(min=1.0)
            elif xyz_loss_type == "CE_coor":
                gt_xyz_bin = gt_xyz_bin.long()
                loss_func = CrossEntropyHeatmapLoss(reduction="sum", weight=None)  # g_head_config.XYZ_BIN+1
                loss_dict["loss_coor_x"] = loss_func(
                    out_x * gt_mask_xyz[:, None], gt_xyz_bin[:, 0] * gt_mask_xyz.long()
                ) / gt_mask_xyz.sum().float().clamp(min=1.0)
                loss_dict["loss_coor_y"] = loss_func(
                    out_y * gt_mask_xyz[:, None], gt_xyz_bin[:, 1] * gt_mask_xyz.long()
                ) / gt_mask_xyz.sum().float().clamp(min=1.0)
                loss_dict["loss_coor_z"] = loss_func(
                    out_z * gt_mask_xyz[:, None], gt_xyz_bin[:, 2] * gt_mask_xyz.long()
                ) / gt_mask_xyz.sum().float().clamp(min=1.0)
            else:
                raise NotImplementedError(f"unknown xyz loss type: {xyz_loss_type}")
            loss_dict["loss_coor_x"] *= loss_config.xyz_lw
            loss_dict["loss_coor_y"] *= loss_config.xyz_lw
            loss_dict["loss_coor_z"] *= loss_config.xyz_lw

        # mask loss ----------------------------------
        if not g_head_config.freeze:
            mask_loss_type = loss_config.mask_loss_type
            gt_mask = gt_masks[loss_config.mask_loss_gt]
            if mask_loss_type == "L1":
                loss_dict["loss_mask"] = nn.L1Loss(reduction="mean")(out_mask[:, 0, :, :], gt_mask)
            elif mask_loss_type == "BCE":
                loss_dict["loss_mask"] = nn.BCEWithLogitsLoss(reduction="mean")(out_mask[:, 0, :, :], gt_mask)
            elif mask_loss_type == "RW_BCE":
                loss_dict["loss_mask"] = weighted_ex_loss_probs(
                    torch.sigmoid(out_mask[:, 0, :, :]), gt_mask, weight=None
                )
            elif mask_loss_type == "dice":
                loss_dict["loss_mask"] = soft_dice_loss(
                    torch.sigmoid(out_mask[:, 0, :, :]), gt_mask, eps=0.002, reduction="mean"
                )
            elif mask_loss_type == "CE":
                loss_dict["loss_mask"] = nn.CrossEntropyLoss(reduction="mean")(out_mask, gt_mask.long())
            else:
                raise NotImplementedError(f"unknown mask loss type: {mask_loss_type}")
            loss_dict["loss_mask"] *= loss_config.mask_lw

        # roi region loss --------------------
        if not g_head_config.freeze:
            region_loss_type = loss_config.region_loss_type
            gt_mask_region = gt_masks[loss_config.region_loss_mask_gt]
            if region_loss_type == "CE":
                gt_region = gt_region.long()
                loss_func = nn.CrossEntropyLoss(reduction="sum", weight=None)  # g_head_config.XYZ_BIN+1
                loss_dict["loss_region"] = loss_func(
                    out_region * gt_mask_region[:, None], gt_region * gt_mask_region.long()
                ) / gt_mask_region.sum().float().clamp(min=1.0)
            else:
                raise NotImplementedError(f"unknown region loss type: {region_loss_type}")
            loss_dict["loss_region"] *= loss_config.region_lw

        # point matching loss ---------------
        if loss_config.pm_lw > 0:
            assert (gt_points is not None) and (gt_trans is not None) and (gt_rot is not None)
            loss_func = PyPMLoss(
                loss_type=loss_config.pm_loss_type,
                beta=loss_config.pm_smooth_l1_beta,
                reduction="mean",
                loss_weight=loss_config.pm_lw,
                norm_by_extent=loss_config.pm_norm_by_extent,
                symmetric=loss_config.pm_loss_sym,
                disentangle_t=loss_config.pm_disentangle_t,
                disentangle_z=loss_config.pm_disentangle_z,
                t_loss_use_points=loss_config.pm_t_use_points,
                r_only=loss_config.pm_r_only,
            )
            loss_pm_dict = loss_func(
                pred_rots=out_rot,
                gt_rots=gt_rot,
                points=gt_points,
                pred_transes=out_trans,
                gt_transes=gt_trans,
                extents=extents,
                sym_infos=sym_infos,
            )
            loss_dict.update(loss_pm_dict)

        # rot_loss ----------
        if loss_config.rot_lw > 0:
            if loss_config.rot_loss_type == "angular":
                loss_dict["loss_rot"] = angular_distance(out_rot, gt_rot)
            elif loss_config.rot_loss_type == "L2":
                loss_dict["loss_rot"] = rot_l2_loss(out_rot, gt_rot)
            else:
                raise ValueError(f"Unknown rot loss type: {loss_config.rot_loss_type}")
            loss_dict["loss_rot"] *= loss_config.rot_lw

        # centroid loss -------------
        if loss_config.centroid_lw > 0:
            assert (
                pnp_net_config.trans_type == "centroid_z"
            ), "centroid loss is only valid for predicting centroid2d_rel_delta"

            if loss_config.centroid_loss_type == "L1":
                loss_dict["loss_centroid"] = nn.L1Loss(reduction="mean")(out_centroid, gt_trans_ratio[:, :2])
            elif loss_config.centroid_loss_type == "L2":
                loss_dict["loss_centroid"] = L2Loss(reduction="mean")(out_centroid, gt_trans_ratio[:, :2])
            elif loss_config.centroid_loss_type == "MSE":
                loss_dict["loss_centroid"] = nn.MSELoss(reduction="mean")(out_centroid, gt_trans_ratio[:, :2])
            else:
                raise ValueError(f"Unknown centroid loss type: {loss_config.centroid_loss_type}")
            loss_dict["loss_centroid"] *= loss_config.centroid_lw

        # z loss ------------------
        if loss_config.z_lw > 0:
            z_type = pnp_net_config.z_type
            if z_type == "REL":
                gt_z = gt_trans_ratio[:, 2]
            elif z_type == "ABS":
                gt_z = gt_trans[:, 2]
            else:
                raise NotImplementedError

            z_loss_type = loss_config.z_loss_type
            if z_loss_type == "L1":
                loss_dict["loss_z"] = nn.L1Loss(reduction="mean")(out_trans_z, gt_z)
            elif z_loss_type == "L2":
                loss_dict["loss_z"] = L2Loss(reduction="mean")(out_trans_z, gt_z)
            elif z_loss_type == "MSE":
                loss_dict["loss_z"] = nn.MSELoss(reduction="mean")(out_trans_z, gt_z)
            else:
                raise ValueError(f"Unknown z loss type: {z_loss_type}")
            loss_dict["loss_z"] *= loss_config.z_lw

        # trans loss ------------------
        if loss_config.trans_lw > 0:
            if loss_config.trans_loss_disentangle:
                # NOTE: disentangle xy/z
                if loss_config.trans_loss_type == "L1":
                    loss_dict["loss_trans_xy"] = nn.L1Loss(reduction="mean")(out_trans[:, :2], gt_trans[:, :2])
                    loss_dict["loss_trans_z"] = nn.L1Loss(reduction="mean")(out_trans[:, 2], gt_trans[:, 2])
                elif loss_config.trans_loss_type == "L2":
                    loss_dict["loss_trans_xy"] = L2Loss(reduction="mean")(out_trans[:, :2], gt_trans[:, :2])
                    loss_dict["loss_trans_z"] = L2Loss(reduction="mean")(out_trans[:, 2], gt_trans[:, 2])
                elif loss_config.trans_loss_type == "MSE":
                    loss_dict["loss_trans_xy"] = nn.MSELoss(reduction="mean")(out_trans[:, :2], gt_trans[:, :2])
                    loss_dict["loss_trans_z"] = nn.MSELoss(reduction="mean")(out_trans[:, 2], gt_trans[:, 2])
                else:
                    raise ValueError(f"Unknown trans loss type: {loss_config.trans_loss_type}")
                loss_dict["loss_trans_xy"] *= loss_config.trans_lw
                loss_dict["loss_trans_z"] *= loss_config.trans_lw
            else:
                if loss_config.trans_loss_type == "L1":
                    loss_dict["loss_trans_LPnP"] = nn.L1Loss(reduction="mean")(out_trans, gt_trans)
                elif loss_config.trans_loss_type == "L2":
                    loss_dict["loss_trans_LPnP"] = L2Loss(reduction="mean")(out_trans, gt_trans)

                elif loss_config.trans_loss_type == "MSE":
                    loss_dict["loss_trans_LPnP"] = nn.MSELoss(reduction="mean")(out_trans, gt_trans)
                else:
                    raise ValueError(f"Unknown trans loss type: {loss_config.trans_loss_type}")
                loss_dict["loss_trans_LPnP"] *= loss_config.trans_lw

        # bind loss (R^T@t)
        if loss_config.get("BIND_LW", 0.0) > 0.0:
            pred_bind = torch.bmm(out_rot.permute(0, 2, 1), out_trans.view(-1, 3, 1)).view(-1, 3)
            gt_bind = torch.bmm(gt_rot.permute(0, 2, 1), gt_trans.view(-1, 3, 1)).view(-1, 3)
            if loss_config.bind_loss_type == "L1":
                loss_dict["loss_bind"] = nn.L1Loss(reduction="mean")(pred_bind, gt_bind)
            elif loss_config.bind_loss_type == "L2":
                loss_dict["loss_bind"] = L2Loss(reduction="mean")(pred_bind, gt_bind)
            elif loss_config.centroid_loss_type == "MSE":
                loss_dict["loss_bind"] = nn.MSELoss(reduction="mean")(pred_bind, gt_bind)
            else:
                raise ValueError(f"Unknown bind loss (R^T@t) type: {loss_config.bind_loss_type}")
            loss_dict["loss_bind"] *= loss_config.bind_lw

        if net_config.use_mtl:
            for _k in loss_dict:
                _name = _k.replace("loss_", "log_var_")
                cur_log_var = getattr(self, _name)
                loss_dict[_k] = loss_dict[_k] * torch.exp(-cur_log_var) + torch.log(1 + torch.exp(cur_log_var))
        return loss_dict

import torch
import io
import torchvision
import numpy as np
import os.path as osp
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functools import partial
from einops import rearrange
from fvcore.nn import smooth_l1_loss
from PIL import Image
from detectron2.layers import paste_masks_in_image
from src.estimator.losses.pm_loss import PyPMLoss
from src.estimator.losses.mask_losses import weighted_ex_loss_probs
from src.estimator.losses.ssim import MS_SSIM
from src.estimator.utils.edge_utils import compute_mask_edge_weights
from src.estimator.utils.data_utils import xyz_to_region_batch, denormalize_image
from src.estimator.utils.zoom_utils import batch_crop_resize
from src.estimator.utils.vis_utils.image import grid_show, heatmap

def batch_data_self(cfg, data, file_names, device="cuda", phase="train"):
    # if phase != "train":
    #     return batch_data_test_self(cfg, data, device=device)
    out_dict = {}
    # batch self training data
    net_cfg = cfg.model.pose_net
    out_res = net_cfg.output_res

    tensor_kwargs = {"dtype": torch.float32, "device": device}
    batch = {}
    for d in data:
      file_names.append(d["file_name"])
    
    # the image, infomation data and data from detection
    batch["image_id"] = torch.stack([torch.from_numpy(np.array(d["image_id"])) for d in data], dim=0).to(device, non_blocking=True)
    # augmented roi_image
    batch["roi_img"] = torch.stack([d["roi_img"] for d in data], dim=0).to(device, non_blocking=True)
    # original roi_image
    batch["roi_gt_img"] = torch.stack([d["roi_gt_img"] for d in data], dim=0).to(device, non_blocking=True)
    # original image
    batch["gt_img"] = torch.stack([d["gt_img"] for d in data], dim=0).to(device, non_blocking=True)
    # synthetic image    
    batch["syn_img"] = torch.stack([d["syn_image"] for d in data], dim=0).to(device, non_blocking=True)
    batch["roi_syn_img"] = torch.stack([d["roi_syn_image"] for d in data], dim=0).to(device, non_blocking=True)
    batch["syn_mask"] = torch.stack([d["syn_mask"] for d in data], dim=0).to(device, non_blocking=True)
    batch["roi_syn_mask"] = torch.stack([d["roi_syn_mask"] for d in data], dim=0).to(device, non_blocking=True)
    im_H, im_W = batch["gt_img"].shape[-2:]

    batch["roi_cls"] = torch.tensor([d["roi_cls"] for d in data], dtype=torch.long).to(device, non_blocking=True)
    bs = batch["roi_cls"].shape[0]
    if "roi_coord_2d" in data[0]:
        batch["roi_coord_2d"] = torch.stack([d["roi_coord_2d"] for d in data], dim=0).to(
            device=device, non_blocking=True
        )
    
    batch["roi_cam"] = torch.stack([d["cam"] for d in data], dim=0).to(device, non_blocking=True)
    batch["roi_center"] = torch.stack([d["bbox_center"] for d in data], dim=0).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    batch["roi_scale"] = torch.tensor([d["scale"] for d in data], device=device, dtype=torch.float32)
    # for crop and resize
    rois_xy0 = batch["roi_center"] - batch["roi_scale"].view(bs, -1) / 2  # bx2
    rois_xy1 = batch["roi_center"] + batch["roi_scale"].view(bs, -1) / 2  # bx2
    batch["inst_rois"] = torch.cat([torch.arange(bs, **tensor_kwargs).view(-1, 1), rois_xy0, rois_xy1], dim=1)

    batch["roi_extent"] = torch.stack([d["roi_extent"] for d in data], dim=0).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    if "sym_info" in data[0]:
        batch["sym_info"] = [d["sym_info"] for d in data]

    if "roi_points" in data[0]:
        batch["roi_points"] = torch.stack([d["roi_points"] for d in data], dim=0).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
    if "roi_fps_points" in data[0]:
        batch["roi_fps_points"] = torch.stack([d["roi_fps_points"] for d in data], dim=0).to(
            device=device, dtype=torch.float32, non_blocking=True
        )

    if cfg.model.pseudo_pose_type == "pose_refine" and "pose_refine" in data[0]:
        batch["pseudo_rot"] = torch.stack([d["pose_refine"][:3, :3] for d in data], dim=0).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        batch["pseudo_trans"] = torch.stack([d["pose_refine"][:3, 3] for d in data], dim=0).to(
            device=device, dtype=torch.float32, non_blocking=True
        )

    batch["roi_wh"] = torch.stack([d["roi_wh"] for d in data], dim=0).to(device, non_blocking=True)
    batch["resize_ratio"] = torch.tensor([d["resize_ratio"] for d in data]).to(
        device=device, dtype=torch.float32, non_blocking=True
    )

    return batch

def compute_self_loss(
        cfg,
        batch,
        pred_rot,
        pred_trans,
        ren,
        ren_models,
        tb_writer=None,
        iteration=None,
):
    loss_config = cfg.train.self_loss
    vis_data = {}
    loss_dict = {}
    vis_i = 0

    im_H, im_W = batch["gt_img"].shape[-2:]
    batch["K_renderer"] = batch["roi_cam"].clone()
    cur_models = [ren_models[int(_l)] for _l in batch["roi_cls"]]
    ren_func = {"batch": partial(ren.render_batch), "batch_tex": partial(ren.render_batch_tex, uv_type="face")}[
        'batch'
    ]

    # Differentiable rendering
    ren_ret = ren_func(
        pred_rot,
        pred_trans,
        cur_models,
        Ks=batch["K_renderer"],
        width=im_W,
        height=im_H,
        mode=["color", "prob"],
    )
    ren_img = rearrange(ren_ret["color"][..., [2, 1, 0]], "b h w c -> b c h w")  # bgr;[0,1]
    ren_mask = rearrange(ren_ret["mask"].to(torch.float32), "b h w -> b 1 h w")
    if "prob" in ren_ret.keys():
        ren_prob = rearrange(ren_ret["prob"].to(torch.float32), "b h w -> b 1 h w")
    else:
        ren_prob = None


    # compute edge weight
    syn_mask = (batch['syn_mask']> 0.5).to(torch.float32)
    if loss_config.mask_weight_type == "edge_lower":
        mask_weight_in_im = compute_mask_edge_weights(
            syn_mask[:, None, :, :], dilate_kernel_size=11, erode_kernel_size=11, edge_lower=True
        )
    else:  # none (default)
        mask_weight_in_im = torch.ones_like(syn_mask[:, None, :, :])

    # Visualization
    if tb_writer is not None:
        gt_img_vis = batch["gt_img"][vis_i].cpu().numpy()
        gt_img_vis = denormalize_image(gt_img_vis, cfg)[::-1].astype("uint8")
        vis_data["gt/image"] = rearrange(gt_img_vis, "c h w -> h w c")

        ren_img_vis = ren_img[vis_i].detach().cpu().numpy()
        ren_img_vis = denormalize_image(ren_img_vis, cfg)[::-1].astype("uint8")
        vis_data["ren/image"] = rearrange(ren_img_vis, "c h w -> h w c")

        pseudo_mask_vis = syn_mask[vis_i].detach().cpu().numpy()
        roi_syn_mask = batch['roi_syn_mask'][vis_i].detach().cpu().numpy()
        vis_data["pseudo/mask_roi"] = pseudo_mask_vis
        vis_data["pseudo/syn_mask_roi"] = roi_syn_mask


        pseudo_syn_image = batch["syn_img"][vis_i].cpu().numpy()
        pseudo_syn_img_vis = denormalize_image(pseudo_syn_image, cfg)[::-1].astype("uint8")
        vis_data["pseduo/syn_image"] = rearrange(pseudo_syn_img_vis, "c h w -> h w c")

        roi_syn_image = batch["roi_syn_img"][vis_i].cpu().numpy()
        roi_syn_img_vis = denormalize_image(roi_syn_image, cfg)[::-1].astype("uint8")
        vis_data["pseduo/roi_syn_image"] = rearrange(roi_syn_img_vis, "c h w -> h w c")


    # Mask loss
    if loss_config.mask_loss_cfg.loss_weight > 0:
        if tb_writer is not None:
            ren_prob_roi = batch_crop_resize(
                ren_prob, batch["inst_rois"], out_H=syn_mask.shape[-2], out_W=syn_mask.shape[-1]
            )
            ren_prob_roi_vis = ren_prob_roi[vis_i].detach().cpu().numpy()
            vis_data["ren/prob_roi"] = ren_prob_roi_vis[0]

            mask_diff_pseudo_ren_vis = heatmap(pseudo_mask_vis[0] - ren_prob_roi_vis[0], to_rgb=True)
            vis_data["diff/mask_pseudo_ren"] = mask_diff_pseudo_ren_vis
        if loss_config.mask_loss_cfg.loss_type == "RW_BCE":
            loss_mask = weighted_ex_loss_probs(
                ren_prob, syn_mask[:, None, :, :], weight=mask_weight_in_im
            )
        else:
            raise ValueError(f"Unknown mask loss type: {loss_config.mask_loss_cfg.loss_type}")
        loss_dict.update({"loss_mask": loss_mask})

    if tb_writer is not None:
        in_res = cfg.model.pose_net.input_res
        ren_img_roi = batch_crop_resize(ren_img, batch["inst_rois"], out_H=in_res, out_W=in_res)
        roi_syn_img = batch["roi_syn_img"]
        pseudo_mask_roi = F.interpolate(syn_mask[:, None, ...], size=(in_res, in_res), mode="nearest")

    # Syn image loss
    if loss_config.syn_rgb_loss_cfg.loss_weight > 0:
        loss_syn_rgb = smooth_l1_loss(roi_syn_img, ren_img_roi, beta=0.2, reduction="sum"
        )/ max(1, pseudo_mask_roi.sum())
        loss_dict["loss_syn_rgb"] = loss_syn_rgb * loss_config.syn_rgb_loss_cfg.loss_weight

    if loss_config.ms_ssim_loss_cfg.loss_weight > 0:
        ms_ssim_func = MS_SSIM(data_range=1.0, normalize=True).cuda()
        loss_ms_ssim_obj = (1 - ms_ssim_func(roi_syn_img, ren_img_roi)).mean()
        loss_dict["loss_ms_ssim"] = loss_ms_ssim_obj * loss_config.ms_ssim_loss_cfg.loss_weight

    # Self pm loss
    if loss_config.pm_loss_cfg.loss_weight > 0:
        pm_loss_func = PyPMLoss(**loss_config.pm_loss_cfg)
        loss_pm_dict = pm_loss_func(
            pred_rots=pred_rot,
            gt_rots=batch['pseudo_rot'],
            points=batch["roi_points"],
            pred_transes=pred_trans,
            gt_transes=batch["pseudo_trans"],
            extents=batch["roi_extent"],
            sym_infos=batch["sym_info"],
        )
        loss_dict.update(loss_pm_dict)

    # grid_show and write to tensorboard -----------------------------------------------------
    if tb_writer is not None:
        # TODO: maybe move this into self_engine and merge with other vis_data
        show_ims = list(vis_data.values())
        show_titles = list(vis_data.keys())
        n_per_col = 5
        nrow = int(np.ceil(len(show_ims) / n_per_col))
        fig = grid_show(show_ims, show_titles, row=nrow, col=n_per_col, show=False)
        im_grid = fig2img(fig)
        plt.close(fig)
        im_grid = torchvision.transforms.ToTensor()(im_grid)
        tb_writer.add_image("vis_im_grid", im_grid, global_step=iteration)

    return loss_dict

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    return img

def get_dibr_models_renderer(cfg, data_ref, obj_names, output_db=True, gpu_id=None):
    """
    Args:
        output_db (bool): Compute and output image-space derivates of barycentrics.
    """
    from lib.dr_utils.dib_renderer_x.renderer_dibr import load_ply_models, Renderer_dibr

    obj_ids = [data_ref.obj2id[_obj] for _obj in obj_names]

    model_scaled_root = data_ref.model_scaled_simple_dir
    obj_paths = [osp.join(model_scaled_root, "{}/textured.obj".format(_obj)) for _obj in obj_names]

    texture_paths = None
    if data_ref.texture_paths is not None:
        texture_paths = [osp.join(model_scaled_root, "{}/texture_map.png".format(_obj)) for _obj in obj_names]

    models = load_ply_models(
        obj_paths=obj_paths,
        texture_paths=texture_paths,
        vertex_scale=data_ref.vertex_scale,
        tex_resize=True,  # to reduce gpu memory usage
        width=512,
        height=512,
    )
    ren_dibr = Renderer_dibr(
        height=cfg.renderer.dibr.height, width=cfg.renderer.dibr.width, mode=cfg.renderer.dibr.mode
    )
    return models, ren_dibr
import copy
import torch
import timm
import logging
import torch.nn as nn
from ..models.GDRN import GDRN
from ..models.necks.fpn import FPN
from ..models.heads.head import TopDownMaskXyzRegionHead, ConvPnPNet
from .common import get_xyz_mask_region_out_dim
from mmcv.runner.checkpoint import load_checkpoint

logger = logging.getLogger(__name__)

def load_model(config):
    """
    Load the model from the given configuration.

    Args:
        config (dict): Configuration dictionary containing model parameters.

    Returns:
        torch.nn.Module: The loaded model.
    """
    model_config = config.model
    backbone_config = model_config.pose_net.backbone

    params_lr_list = []

    # Load backbone
    init_backbone_args = copy.deepcopy(backbone_config.init_config)
    backbone_type = init_backbone_args.pop('type')
    if 'timm/' in backbone_type:
        init_backbone_args['model_name'] = backbone_type.split('timm/')[-1]
    
    init_backbone_args.out_indices = tuple(init_backbone_args.pop('out_indices'))
    backbone = timm.create_model(**init_backbone_args)
    params_lr_list.append(
            {"params": filter(lambda p: p.requires_grad, backbone.parameters()), "lr": float(config.train.base_lr)}
        )
    
    # Load neck if enabled
    if model_config.pose_net.neck.enable:
        neck, neck_params = get_neck(config)
    else:
        neck, neck_params = None, []
    
    # Load geo head
    geo_head, geo_head_params = get_geo_head(config)
    params_lr_list.extend(geo_head_params)

    # Load pnp net
    pnp_net, pnp_net_params = get_pnp_net(config)
    params_lr_list.extend(pnp_net_params)
    
    # Build student model and teacher model
    model = GDRN(config, backbone, neck=neck, geo_head_net=geo_head, pnp_net=pnp_net)
    
    if config.model.pretrained:
        logger.info(f"Loading pretrained weights from {config.model.pretrained}")
        load_checkpoint(model, config.model.pretrained, logger=logger)
    
    return model, params_lr_list

def get_geo_head(config):
    net_config = config.model.pose_net
    geo_head_config = net_config.geo_head
    params_lr_list = []

    geo_head_init_config = copy.deepcopy(geo_head_config.init_config)
    geo_head_type = geo_head_init_config.pop("type")

    xyz_num_classes = net_config.num_classes if geo_head_config.xyz_class_aware else 1
    mask_num_classes = net_config.num_classes if geo_head_config.mask_class_aware else 1
    
    xyz_dim, mask_dim, region_dim = get_xyz_mask_region_out_dim(config)
    region_num_classes = net_config.num_classes if geo_head_config.region_class_aware else 1
    geo_head_init_config.update(
        xyz_num_classes=xyz_num_classes,
        mask_num_classes=mask_num_classes,
        region_num_classes=region_num_classes,
        xyz_out_dim=xyz_dim,
        mask_out_dim=mask_dim,
        region_out_dim=region_dim,
    )

    if geo_head_type == 'TopDownMaskXyzRegionHead':
        geo_head = TopDownMaskXyzRegionHead(**geo_head_init_config)

    if geo_head_config.freeze:
        for param in geo_head.parameters():
            with torch.no_grad():
                param.requires_grad = False
    else:
        params_lr_list.append(
            {
                "params": filter(lambda p: p.requires_grad, geo_head.parameters()),
                "lr": float(config.train.base_lr) * geo_head_config.lr_mult,
            }
        )

    return geo_head, params_lr_list
    
def get_neck(config):
    net_config = config.model.posenet
    neck_config = net_config.NECK
    params_lr_list = []
    if neck_config.ENABLED:
        neck_init_config = copy.deepcopy(neck_config.INIT_config)
        neck_type = neck_init_config.pop("type")
        neck = FPN[neck_type](**neck_init_config)
        if neck_config.FREEZE:
            for param in neck.parameters():
                with torch.no_grad():
                    param.requires_grad = False
        else:
            params_lr_list.append(
                {
                    "params": filter(lambda p: p.requires_grad, neck.parameters()),
                    "lr": float(config.SOLVER.BASE_LR) * neck_config.LR_MULT,
                }
            )
    else:
        neck = None
    return neck, params_lr_list

def get_pnp_net(config):
    net_config = config.model.pose_net
    g_head_config = net_config.geo_head
    pnp_net_config = net_config.pnp_net
    loss_config = config.train.loss

    xyz_dim, mask_dim, region_dim = get_xyz_mask_region_out_dim(config)

    if loss_config.xyz_loss_type in ["CE_coor", "CE"]:
        pnp_net_in_channel = xyz_dim - 3  # for bin xyz, no bg channel
    else:
        pnp_net_in_channel = xyz_dim

    if pnp_net_config.with_2d_coord:
        pnp_net_in_channel += 2

    if pnp_net_config.region_attention:
        pnp_net_in_channel += g_head_config.num_regions

    if pnp_net_config.mask_attention in ["concat"]:  # do not add dim for none/mul
        pnp_net_in_channel += 1

    if pnp_net_config.rot_type in ["allo_quat", "ego_quat"]:
        rot_dim = 4
    elif pnp_net_config.rot_type in [
        "allo_log_quat",
        "ego_log_quat",
        "allo_lie_vec",
        "ego_lie_vec",
    ]:
        rot_dim = 3
    elif pnp_net_config.rot_type in ["allo_rot6d", "ego_rot6d"]:
        rot_dim = 6
    else:
        raise ValueError(f"Unknown ROT_TYPE: {pnp_net_config.rot_type}")

    pnp_net_init_config = copy.deepcopy(pnp_net_config.init_config)
    pnp_head_type = pnp_net_init_config.pop("type")

    if pnp_head_type == 'ConvPnPNet':
        pnp_net_init_config.update(
            nIn=pnp_net_in_channel,
            rot_dim=rot_dim,
            num_regions=g_head_config.num_regions,
            mask_attention_type=pnp_net_config.mask_attention,
        )
        pnp_net = ConvPnPNet(**pnp_net_init_config)

    params_lr_list = []
    if pnp_net_config.freeze:
        for param in pnp_net.parameters():
            with torch.no_grad():
                param.requires_grad = False
    else:
        if pnp_net_config.train_r_only:
            logger.info("Train fc_r only...")
            for name, param in pnp_net.named_parameters():
                if "fc_r" not in name:
                    with torch.no_grad():
                        param.requires_grad = False
        params_lr_list.append(
            {
                "params": filter(lambda p: p.requires_grad, pnp_net.parameters()),
                "lr": float(config.train.base_lr) * pnp_net_config.lr_mult,
            }
        )
    return pnp_net, params_lr_list
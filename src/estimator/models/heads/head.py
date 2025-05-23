import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import normal_init, constant_init
from ...utils.torch_utils.layers.layer_utils import get_norm, get_nn_act_func
from ...utils.torch_utils.layers.conv_module import ConvModule
from ...utils.torch_utils.layers.dropblock import DropBlock2D, LinearScheduler


class TopDownMaskXyzRegionHead(nn.Module):
    def __init__(
        self,
        in_dim,
        up_types=("deconv", "bilinear", "bilinear"),
        deconv_kernel_size=3,
        num_conv_per_block=2,
        feat_dim=256,
        feat_kernel_size=3,
        norm="GN",
        num_gn_groups=32,
        act="GELU",
        out_kernel_size=1,
        out_layer_shared=True,
        mask_num_classes=1,
        xyz_num_classes=1,
        region_num_classes=1,
        mask_out_dim=1,
        xyz_out_dim=3,
        region_out_dim=65,  # 64+1
    ):
        """
        Args:
            up_types: use up-conv or deconv for each up-sampling layer
                ("bilinear", "bilinear", "bilinear")
                ("deconv", "bilinear", "bilinear")  # CDPNv2 rot head
                ("deconv", "deconv", "deconv")  # CDPNv1 rot head
                ("nearest", "nearest", "nearest")  # implement here but maybe won't use
        NOTE: default from stride 32 to stride 4 (3 ups)
        """
        super().__init__()
        assert out_kernel_size in [
            1,
            3,
        ], "Only support output kernel size: 1 and 3"
        assert deconv_kernel_size in [
            1,
            3,
            4,
        ], "Only support deconv kernel size: 1, 3, and 4"
        assert len(up_types) > 0, up_types

        self.features = nn.ModuleList()
        for i, up_type in enumerate(up_types):
            _in_dim = in_dim if i == 0 else feat_dim
            if up_type == "deconv":
                (
                    deconv_kernel,
                    deconv_pad,
                    deconv_out_pad,
                ) = _get_deconv_pad_outpad(deconv_kernel_size)
                self.features.append(
                    nn.ConvTranspose2d(
                        _in_dim,
                        feat_dim,
                        kernel_size=deconv_kernel,
                        stride=2,
                        padding=deconv_pad,
                        output_padding=deconv_out_pad,
                        bias=False,
                    )
                )
                self.features.append(get_norm(norm, feat_dim, num_gn_groups=num_gn_groups))
                self.features.append(get_nn_act_func(act))
            elif up_type == "bilinear":
                self.features.append(nn.UpsamplingBilinear2d(scale_factor=2))
            elif up_type == "nearest":
                self.features.append(nn.UpsamplingNearest2d(scale_factor=2))
            else:
                raise ValueError(f"Unknown up_type: {up_type}")

            if up_type in ["bilinear", "nearest"]:
                assert num_conv_per_block >= 1, num_conv_per_block
            for i_conv in range(num_conv_per_block):
                if i == 0 and i_conv == 0 and up_type in ["bilinear", "nearest"]:
                    conv_in_dim = in_dim
                else:
                    conv_in_dim = feat_dim
                self.features.append(
                    ConvModule(
                        conv_in_dim,
                        feat_dim,
                        kernel_size=feat_kernel_size,
                        padding=(feat_kernel_size - 1) // 2,
                        norm=norm,
                        num_gn_groups=num_gn_groups,
                        act=act,
                    )
                )

        self.out_layer_shared = out_layer_shared
        self.mask_num_classes = mask_num_classes
        self.xyz_num_classes = xyz_num_classes
        self.region_num_classes = region_num_classes

        self.mask_out_dim = mask_out_dim
        self.xyz_out_dim = xyz_out_dim
        self.region_out_dim = region_out_dim

        if self.out_layer_shared:
            out_dim = (
                self.mask_out_dim * self.mask_num_classes
                + self.xyz_out_dim * self.xyz_num_classes
                + self.region_out_dim * self.region_num_classes
            )
            self.out_layer = nn.Conv2d(
                feat_dim,
                out_dim,
                kernel_size=out_kernel_size,
                padding=(out_kernel_size - 1) // 2,
                bias=True,
            )
        else:
            self.mask_out_layer = nn.Conv2d(
                feat_dim,
                self.mask_out_dim * self.mask_num_classes,
                kernel_size=out_kernel_size,
                padding=(out_kernel_size - 1) // 2,
                bias=True,
            )
            self.xyz_out_layer = nn.Conv2d(
                feat_dim,
                self.xyz_out_dim * self.xyz_num_classes,
                kernel_size=out_kernel_size,
                padding=(out_kernel_size - 1) // 2,
                bias=True,
            )
            self.region_out_layer = nn.Conv2d(
                feat_dim,
                self.region_out_dim * self.region_num_classes,
                kernel_size=out_kernel_size,
                padding=(out_kernel_size - 1) // 2,
                bias=True,
            )

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
        # init output layers
        if self.out_layer_shared:
            normal_init(self.out_layer, std=0.01)
        else:
            normal_init(self.mask_out_layer, std=0.01)
            normal_init(self.xyz_out_layer, std=0.01)
            normal_init(self.region_out_layer, std=0.01)

    def forward(self, x):
        if isinstance(x, (tuple, list)) and len(x) == 1:
            x = x[0]
        for i, l in enumerate(self.features):
            x = l(x)
        if self.out_layer_shared:
            out = self.out_layer(x)
            mask_dim = self.mask_out_dim * self.mask_num_classes
            mask = out[:, :mask_dim, :, :]

            xyz_dim = self.xyz_out_dim * self.xyz_num_classes
            xyz = out[:, mask_dim : mask_dim + xyz_dim, :, :]

            region = out[:, mask_dim + xyz_dim :, :, :]

            bs, c, h, w = xyz.shape
            xyz = xyz.view(bs, 3, xyz_dim // 3, h, w)
            coor_x = xyz[:, 0, :, :, :]
            coor_y = xyz[:, 1, :, :, :]
            coor_z = xyz[:, 2, :, :, :]

        else:
            mask = self.mask_out_layer(x)

            xyz = self.xyz_out_layer(x)
            bs, c, h, w = xyz.shape
            xyz = xyz.view(bs, 3, c // 3, h, w)
            coor_x = xyz[:, 0, :, :, :]
            coor_y = xyz[:, 1, :, :, :]
            coor_z = xyz[:, 2, :, :, :]

            region = self.region_out_layer(x)
        return mask, coor_x, coor_y, coor_z, region


def _get_deconv_pad_outpad(deconv_kernel):
    """Get padding and out padding for deconv layers."""
    if deconv_kernel == 4:
        padding = 1
        output_padding = 0
    elif deconv_kernel == 3:
        padding = 1
        output_padding = 1
    elif deconv_kernel == 2:
        padding = 0
        output_padding = 0
    else:
        raise ValueError(f"Not supported num_kernels ({deconv_kernel}).")

    return deconv_kernel, padding, output_padding

class ConvPnPNet(nn.Module):
    def __init__(
        self,
        nIn,
        num_regions=8,
        mask_attention_type="none",
        featdim=128,
        rot_dim=6,
        num_stride2_layers=3,
        num_extra_layers=0,
        norm="GN",
        num_gn_groups=32,
        act="relu",
        drop_prob=0.0,
        dropblock_size=5,
        flat_op="flatten",
        final_spatial_size=(8, 8),
        denormalize_by_extent=True,
    ):
        """
        Args:
            nIn: input feature channel
            spatial_pooltype: max | soft
            spatial_topk: 1
            flat_op: flatten | avg | avg-max | avg-max-min
        """
        super().__init__()
        self.featdim = featdim
        self.num_regions = num_regions
        self.mask_attention_type = mask_attention_type
        self.flat_op = flat_op
        self.denormalize_by_extent = denormalize_by_extent

        conv_act = get_nn_act_func(act)
        if act == "relu":
            self.act = get_nn_act_func("lrelu")  # legacy model
        else:
            self.act = get_nn_act_func(act)
        # -----------------------------------
        self.drop_prob = drop_prob
        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=drop_prob, block_size=dropblock_size),
            start_value=0.0,
            stop_value=drop_prob,
            nr_steps=5000,
        )

        self.features = nn.ModuleList()
        for i in range(num_stride2_layers):
            _in_channels = nIn if i == 0 else featdim
            self.features.append(
                nn.Conv2d(
                    _in_channels,
                    featdim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            self.features.append(get_norm(norm, featdim, num_gn_groups=num_gn_groups))
            self.features.append(conv_act)
        for i in range(num_extra_layers):
            self.features.append(
                nn.Conv2d(
                    featdim,
                    featdim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            )
            self.features.append(get_norm(norm, featdim, num_gn_groups=num_gn_groups))
            self.features.append(conv_act)

        final_h, final_w = final_spatial_size
        fc_in_dim = {
            "flatten": featdim * final_h * final_w,
            "avg": featdim,
            "avg-max": featdim * 2,
            "avg-max-min": featdim * 3,
        }[flat_op]

        # self.fc1 = nn.Linear(featdim * 8 * 8 + 128, 1024)  # NOTE: 128 for extents feature
        self.fc1 = nn.Linear(fc_in_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc_r = nn.Linear(256, rot_dim)  # quat or rot6d
        # TODO: predict centroid and z separately
        self.fc_t = nn.Linear(256, 3)

        # feature for extent
        # self.extent_fc1 = nn.Linear(3, 64)
        # self.extent_fc2 = nn.Linear(64, 128)

        # init ------------------------------------
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.001)
        normal_init(self.fc_r, std=0.01)
        normal_init(self.fc_t, std=0.01)

    def forward(self, coor_feat, region=None, extents=None, mask_attention=None):
        """
        Args:
             since this is the actual correspondence
            x: (B,C,H,W)
            extents: (B, 3)
        Returns:

        """
        bs, in_c, fh, fw = coor_feat.shape
        if in_c in [3, 5] and self.denormalize_by_extent and extents is not None:
            coor_feat[:, :3, :, :] = (coor_feat[:, :3, :, :] - 0.5) * extents.view(bs, 3, 1, 1)
        # convs
        if region is not None:
            x = torch.cat([coor_feat, region], dim=1)
        else:
            x = coor_feat

        if self.mask_attention_type != "none":
            assert mask_attention is not None
            if self.mask_attention_type == "mul":
                x = x * mask_attention
            elif self.mask_attention_type == "concat":
                x = torch.cat([x, mask_attention], dim=1)
            else:
                raise ValueError(f"Wrong mask attention type: {self.mask_attention_type}")

        if self.drop_prob > 0:
            self.dropblock.step()  # increment number of iterations
            x = self.dropblock(x)

        for _i, layer in enumerate(self.features):
            x = layer(x)

        flat_conv_feat = x.flatten(2)  # [B,featdim,*]
        if self.flat_op == "flatten":
            flat_conv_feat = flat_conv_feat.flatten(1)
        elif self.flat_op == "avg":
            flat_conv_feat = flat_conv_feat.mean(-1)  # spatial global average pooling
        elif self.flat_op == "avg-max":
            flat_conv_feat = torch.cat([flat_conv_feat.mean(-1), flat_conv_feat.max(-1)[0]], dim=-1)
        elif self.flat_op == "avg-max-min":
            flat_conv_feat = torch.cat(
                [
                    flat_conv_feat.mean(-1),
                    flat_conv_feat.max(-1)[0],
                    flat_conv_feat.min(-1)[0],
                ],
                dim=-1,
            )
        else:
            raise ValueError(f"Invalid flat_op: {self.flat_op}")
        # extent feature
        # # TODO: use extent the other way: denormalize coords
        # x_extent = self.act(self.extent_fc1(extents))
        # x_extent = self.act(self.extent_fc2(x_extent))
        # x = torch.cat([x, x_extent], dim=1)
        #
        x = self.act(self.fc1(flat_conv_feat))
        x = self.act(self.fc2(x))
        #
        rot = self.fc_r(x)
        t = self.fc_t(x)
        return rot, t
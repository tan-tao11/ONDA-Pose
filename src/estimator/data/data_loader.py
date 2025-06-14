# -*- coding: utf-8 -*-
import copy
import logging
import os.path as osp
import pickle
import os
import cv2
import mmcv
import numpy as np
import ref
import torch
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BoxMode
from detectron2.utils.logger import log_first_n
from detectron2.data import get_detection_dataset_dicts

from .base_data_loader import Base_DatasetFromList
from ..utils.data_utils import crop_resize_by_warp_affine, get_2d_coord_np, read_image_mmcv, xyz_to_region
from ..utils.dataset_utils import (
    filter_empty_dets,
    filter_invalid_in_dataset_dicts,
    flat_dataset_dicts,
    my_build_batch_data_loader,
    trivial_batch_collator,
    load_detections_into_dataset,
)
from .my_distributed_sampler import InferenceSampler, RepeatFactorTrainingSampler, TrainingSampler, TrainingSampler_Pair

from ..utils.pysixd import inout, misc
from ..utils.mask_utils import cocosegm2mask, get_edge

from .dataset_factory import register_datasets

from ..utils.smoc_utils import load_camera_pose
from ..utils.ssd_color_transform import ColorAugSSDTransform

logger = logging.getLogger(__name__)


def transform_instance_annotations(
    annotation, transforms, image_size, *, keypoint_hflip_indices=None, bbox_key="bbox_est"
):
    """
    NOTE: Adapted from detection_utils.
    Apply transforms to box, segmentation, keypoints, etc. of annotations of a single instance.

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
        transforms (TransformList):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.

    Returns:
        dict:
            the same input dict with fields bbox_key, "segmentation", "keypoints"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    im_H, im_W = image_size
    bbox = BoxMode.convert(annotation[bbox_key], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # Note that bbox is 1d (per-instance bounding box)
    annotation[bbox_key] = np.array(transforms.apply_box([bbox])[0])
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    if "segmentation" in annotation:
        # NOTE: here we transform segms to binary masks (interp is nearest by default)
        mask = transforms.apply_segmentation(cocosegm2mask(annotation["segmentation"], h=im_H, w=im_W))
        annotation["segmentation"] = mask

    if "keypoints" in annotation:
        keypoints = utils.transform_keypoint_annotations(
            annotation["keypoints"], transforms, image_size, keypoint_hflip_indices
        )
        annotation["keypoints"] = keypoints

    if "centroid_2d" in annotation:
        annotation["centroid_2d"] = transforms.apply_coords(np.array(annotation["centroid_2d"]).reshape(1, 2)).flatten()

    return annotation


def build_gdrn_augmentation(cfg, is_train):
    """Create a list of :class:`Augmentation` from config. when training 6d
    pose, cannot flip.

    Returns:
        list[Augmentation]
    """
    if is_train:
        min_size = cfg.input.min_size_train
        max_size = cfg.input.max_size_train
        sample_style = cfg.input.min_size_train_sampling
    else:
        min_size = cfg.input.min_size_test
        max_size = cfg.input.max_size_test
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    augmentation = []
    augmentation.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        # augmentation.append(T.RandomFlip())
        logger.info("Augmentations used in training: " + str(augmentation))
    return augmentation

class GDRN_DatasetFromList(Base_DatasetFromList):
    """NOTE: we can also use the default DatasetFromList and
    implement a similar custom DataMapper,
    but it is harder to implement some features relying on other dataset dicts
    # https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/common.py
    Wrap a list to a torch Dataset. It produces elements of the list as data.
    """

    def __init__(
        self,
        cfg,
        split,
        lst: list,
        copy: bool = True,
        serialize: bool = True,
        flatten=True,
    ):
        """
        Args:
            lst (list): a list which contains elements to produce.
            copy (bool): whether to deepcopy the element when producing it,
                so that the result can be modified in place without affecting the
                source in the list.
            serialize (bool): whether to hold memory using serialized objects, when
                enabled, data loader workers can use shared RAM from master
                process instead of making a copy.
        """
        self.augmentation = build_gdrn_augmentation(cfg, is_train=(split == "train"))
        if cfg.input.color_aug_prob > 0 and cfg.input.color_aug_type.lower() == "ssd":
            self.augmentation.append(ColorAugSSDTransform(img_format=cfg.input.format))
            logging.getLogger(__name__).info("Color augmentation used in training: " + str(self.augmentation[-1]))
        # fmt: off
        self.img_format = cfg.input.format  # default bgr
        self.with_depth = cfg.input.with_depth
        self.aug_depth = cfg.input.aug_depth
        # note: color augmentation config
        self.color_aug_prob = cfg.input.color_aug_prob
        self.color_aug_type = cfg.input.color_aug_type
        self.color_aug_code = cfg.input.color_aug_code
        # fmt: on
        self.cfg = cfg
        self.split = split  # train | val | test
        if split == "train" and self.color_aug_prob > 0:
            self.color_augmentor = self._get_color_augmentor(aug_type=self.color_aug_type, aug_code=self.color_aug_code)
        else:
            self.color_augmentor = None
        # ------------------------
        # common model infos
        self.fps_points = {}
        self.model_points = {}
        self.extents = {}
        self.sym_infos = {}
        # ----------------------------------------------------
        self.flatten = flatten
        self._lst = flat_dataset_dicts(lst) if flatten else lst
        # ----------------------------------------------------
        self._copy = copy
        self._serialize = serialize

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        if self._serialize:
            logger.info("Serializing {} elements to byte tensors and concatenating them all ...".format(len(self._lst)))
            self._lst = [_serialize(x) for x in self._lst]
            self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
            self._addr = np.cumsum(self._addr)
            self._lst = np.concatenate(self._lst)
            logger.info("Serialized dataset takes {:.2f} MiB".format(len(self._lst) / 1024 ** 2))

    def __len__(self):
        if self._serialize:
            return len(self._addr)
        else:
            return len(self._lst)

    def _get_fps_points(self, dataset_name, with_center=False):
        """convert to label based keys.

        # TODO: get models info similarly
        """
        if dataset_name in self.fps_points:
            return self.fps_points[dataset_name]

        dset_meta = MetadataCatalog.get(dataset_name)
        ref_key = dset_meta.ref_key
        data_ref = ref.__dict__[ref_key]
        objs = dset_meta.objs
        cfg = self.cfg
        num_fps_points = cfg.model.pose_net.geo_head.num_regions
        cur_fps_points = {}
        loaded_fps_points = data_ref.get_fps_points()
        for i, obj_name in enumerate(objs):
            obj_id = data_ref.obj2id[obj_name]
            if with_center:
                cur_fps_points[i] = loaded_fps_points[str(obj_id)][f"fps{num_fps_points}_and_center"]
            else:
                cur_fps_points[i] = loaded_fps_points[str(obj_id)][f"fps{num_fps_points}_and_center"][:-1]
        self.fps_points[dataset_name] = cur_fps_points
        return self.fps_points[dataset_name]

    def _get_model_points(self, dataset_name):
        """convert to label based keys."""
        if dataset_name in self.model_points:
            return self.model_points[dataset_name]

        dset_meta = MetadataCatalog.get(dataset_name)
        ref_key = dset_meta.ref_key
        data_ref = ref.__dict__[ref_key]
        objs = dset_meta.objs
        cfg = self.cfg

        cur_model_points = {}
        num = np.inf
        for i, obj_name in enumerate(objs):
            obj_id = data_ref.obj2id[obj_name]
            model_path = osp.join(data_ref.model_dir, f"obj_{obj_id:06d}.ply")
            model = inout.load_ply(model_path, vertex_scale=data_ref.vertex_scale)
            cur_model_points[i] = pts = model["pts"]
            if pts.shape[0] < num:
                num = pts.shape[0]

        num = min(num, cfg.model.pose_net.loss_cfg.num_pm_points)
        for i in range(len(cur_model_points)):
            keep_idx = np.arange(num)
            np.random.shuffle(keep_idx)  # random sampling
            cur_model_points[i] = cur_model_points[i][keep_idx, :]

        self.model_points[dataset_name] = cur_model_points
        return self.model_points[dataset_name]

    def _get_extents(self, dataset_name):
        """label based keys."""
        if dataset_name in self.extents:
            return self.extents[dataset_name]

        dset_meta = MetadataCatalog.get(dataset_name)
        try:
            ref_key = dset_meta.ref_key
        except:
            # FIXME: for some reason, in distributed training, this need to be re-registered
            register_datasets([dataset_name])
            dset_meta = MetadataCatalog.get(dataset_name)
            ref_key = dset_meta.ref_key

        data_ref = ref.__dict__[ref_key]
        objs = dset_meta.objs
        cfg = self.cfg

        cur_extents = {}
        for i, obj_name in enumerate(objs):
            obj_id = data_ref.obj2id[obj_name]
            model_path = osp.join(data_ref.model_dir, f"obj_{obj_id:06d}.ply")
            model = inout.load_ply(model_path, vertex_scale=data_ref.vertex_scale)
            pts = model["pts"]
            xmin, xmax = np.amin(pts[:, 0]), np.amax(pts[:, 0])
            ymin, ymax = np.amin(pts[:, 1]), np.amax(pts[:, 1])
            zmin, zmax = np.amin(pts[:, 2]), np.amax(pts[:, 2])
            size_x = xmax - xmin
            size_y = ymax - ymin
            size_z = zmax - zmin
            cur_extents[i] = np.array([size_x, size_y, size_z], dtype="float32")

        self.extents[dataset_name] = cur_extents
        return self.extents[dataset_name]

    def _get_sym_infos(self, dataset_name):
        """label based keys."""
        if dataset_name in self.sym_infos:
            return self.sym_infos[dataset_name]

        dset_meta = MetadataCatalog.get(dataset_name)
        ref_key = dset_meta.ref_key
        data_ref = ref.__dict__[ref_key]
        objs = dset_meta.objs
        cfg = self.cfg

        cur_sym_infos = {}
        loaded_models_info = data_ref.get_models_info()
        for i, obj_name in enumerate(objs):
            obj_id = data_ref.obj2id[obj_name]
            model_info = loaded_models_info[str(obj_id)]
            if "symmetries_discrete" in model_info or "symmetries_continuous" in model_info:
                sym_transforms = misc.get_symmetry_transformations(model_info, max_sym_disc_step=0.01)
                sym_info = np.array([sym["R"] for sym in sym_transforms], dtype=np.float32)
            else:
                sym_info = None
            cur_sym_infos[i] = sym_info

        self.sym_infos[dataset_name] = cur_sym_infos
        return self.sym_infos[dataset_name]

    def read_data(self, dataset_dict):
        """load image and annos random shift & scale bbox; crop, rescale."""
        cfg = self.cfg
        net_cfg = cfg.model.pose_net
        g_head_cfg = net_cfg.geo_head
        pnp_net_cfg = net_cfg.pnp_net
        loss_cfg = cfg.train.loss

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        dataset_name = dataset_dict["dataset_name"]

        image = read_image_mmcv(dataset_dict["file_name"], format=self.img_format)
        # should be consistent with the size in dataset_dict
        utils.check_image_size(dataset_dict, image)
        im_H_ori, im_W_ori = image.shape[:2]

        # currently only replace bg for train ###############################
        if self.split == "train":
            # some synthetic data already has bg, img_type should be real or something else but not syn
            img_type = dataset_dict.get("img_type", "real")
            if img_type == "syn":
                log_first_n(logging.WARNING, "replace bg", n=10)
                assert "segmentation" in dataset_dict["inst_infos"]
                mask = cocosegm2mask(
                    dataset_dict["inst_infos"]["segmentation"],
                    im_H_ori,
                    im_W_ori,
                )
                image, mask_trunc = self.replace_bg(
                    image.copy(),
                    mask,
                    return_mask=True,
                    truncate_fg=cfg.input.get("TRUNCATE_FG", False),
                )
            else:  # real image
                if np.random.rand() < cfg.input.change_bg_prob:
                    log_first_n(logging.WARNING, "replace bg for real", n=10)
                    assert "segmentation" in dataset_dict["inst_infos"]
                    mask = cocosegm2mask(
                        dataset_dict["inst_infos"]["segmentation"],
                        im_H_ori,
                        im_W_ori,
                    )
                    image, mask_trunc = self.replace_bg(
                        image.copy(),
                        mask,
                        return_mask=True,
                        truncate_fg=cfg.input.get("TRUNCATE_FG", False),
                    )
                else:
                    mask_trunc = None

        # NOTE: maybe add or change color augment here ===================================
        if self.split == "train" and self.color_aug_prob > 0 and self.color_augmentor is not None:
            if np.random.rand() < self.color_aug_prob:
                if cfg.input.color_aug_syn_only and img_type not in ["real"]:
                    image = self._color_aug(image, self.color_aug_type)
                else:
                    image = self._color_aug(image, self.color_aug_type)

        # other transforms (mainly geometric ones);
        # for 6d pose task, flip is not allowed in general except for some 2d keypoints methods
        image, transforms = T.apply_augmentations(self.augmentation, image)
        im_H, im_W = image_shape = image.shape[:2]  # h, w

        # NOTE: scale camera intrinsic if necessary ================================
        scale_x = im_W / im_W_ori
        scale_y = im_H / im_H_ori  # NOTE: generally scale_x should be equal to scale_y
        if "cam" in dataset_dict:
            if im_W != im_W_ori or im_H != im_H_ori:
                dataset_dict["cam"][0] *= scale_x
                dataset_dict["cam"][1] *= scale_y
            K = dataset_dict["cam"].astype("float32")
            dataset_dict["cam"] = torch.as_tensor(K)

        input_res = net_cfg.input_res
        out_res = net_cfg.output_res

        # CHW -> HWC
        coord_2d = get_2d_coord_np(im_W, im_H, low=0, high=1).transpose(1, 2, 0)

        #################################################################################
        if self.split != "train":
            # don't load annotations at test time
            test_bbox_type = cfg.test.test_bbox_type
            if test_bbox_type == "gt":
                bbox_key = "bbox"
            else:
                bbox_key = f"bbox_{test_bbox_type}"
            assert not self.flatten, "Do not use flattened dicts for test!"
            # here get batched rois
            roi_infos = {}
            # yapf: disable
            roi_keys = ["scene_im_id", "file_name", "cam", "im_H", "im_W",
                        "roi_img", "inst_id", "roi_coord_2d", "roi_coord_2d_rel",
                        "roi_cls", "score", "time", "roi_extent",
                        bbox_key, "bbox_mode", "bbox_center", "roi_wh",
                        "scale", "resize_ratio", "model_info",
            ]
            for _key in roi_keys:
                roi_infos[_key] = []
            # yapf: enable
            # TODO: how to handle image without detections
            #   filter those when load annotations or detections, implement a function for this
            # "annotations" means detections
            for inst_i, inst_infos in enumerate(dataset_dict["annotations"]):
                # inherent image-level infos
                roi_infos["scene_im_id"].append(dataset_dict["scene_im_id"])
                roi_infos["file_name"].append(dataset_dict["file_name"])
                roi_infos["im_H"].append(im_H)
                roi_infos["im_W"].append(im_W)
                roi_infos["cam"].append(dataset_dict["cam"].cpu().numpy())

                # roi-level infos
                roi_infos["inst_id"].append(inst_i)
                roi_infos["model_info"].append(inst_infos["model_info"])

                roi_cls = inst_infos["category_id"]
                roi_infos["roi_cls"].append(roi_cls)
                roi_infos["score"].append(inst_infos.get("score", 1.0))

                roi_infos["time"].append(inst_infos.get("time", 0))

                # extent
                roi_extent = self._get_extents(dataset_name)[roi_cls]
                roi_infos["roi_extent"].append(roi_extent)

                bbox = BoxMode.convert(
                    inst_infos[bbox_key],
                    inst_infos["bbox_mode"],
                    BoxMode.XYXY_ABS,
                )
                bbox = np.array(transforms.apply_box([bbox])[0])
                roi_infos[bbox_key].append(bbox)
                roi_infos["bbox_mode"].append(BoxMode.XYXY_ABS)
                x1, y1, x2, y2 = bbox
                bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
                bw = max(x2 - x1, 1)
                bh = max(y2 - y1, 1)
                scale = max(bh, bw) * cfg.input.dzi_pad_scale
                scale = min(scale, max(im_H, im_W)) * 1.0

                roi_infos["bbox_center"].append(bbox_center.astype("float32"))
                roi_infos["scale"].append(scale)
                roi_wh = np.array([bw, bh], dtype=np.float32)
                roi_infos["roi_wh"].append(roi_wh)
                roi_infos["resize_ratio"].append(out_res / scale)

                # CHW, float32 tensor
                # roi_image
                roi_img = crop_resize_by_warp_affine(
                    image,
                    bbox_center,
                    scale,
                    input_res,
                    interpolation=cv2.INTER_LINEAR,
                ).transpose(2, 0, 1)

                roi_img = self.normalize_image(cfg, roi_img)
                roi_infos["roi_img"].append(roi_img.astype("float32"))

                # roi_coord_2d
                roi_coord_2d = crop_resize_by_warp_affine(
                    coord_2d,
                    bbox_center,
                    scale,
                    out_res,
                    interpolation=cv2.INTER_LINEAR,
                ).transpose(
                    2, 0, 1
                )  # HWC -> CHW
                roi_infos["roi_coord_2d"].append(roi_coord_2d.astype("float32"))

                # roi_coord_2d_rel
                roi_coord_2d_rel = (
                    bbox_center.reshape(2, 1, 1) - roi_coord_2d * np.array([im_W, im_H]).reshape(2, 1, 1)
                ) / scale
                roi_infos["roi_coord_2d_rel"].append(roi_coord_2d_rel.astype("float32"))

            for _key in roi_keys:
                if _key in ["roi_img", "roi_coord_2d", "roi_coord_2d_rel"]:
                    dataset_dict[_key] = torch.as_tensor(roi_infos[_key]).contiguous()
                elif _key in ["model_info", "scene_im_id", "file_name"]:
                    # can not convert to tensor
                    dataset_dict[_key] = roi_infos[_key]
                else:
                    dataset_dict[_key] = torch.tensor(roi_infos[_key])

            return dataset_dict
        #######################################################################################
        # NOTE: currently assume flattened dicts for train
        assert self.flatten, "Only support flattened dicts for train now"
        inst_infos = dataset_dict.pop("inst_infos")
        dataset_dict["roi_cls"] = roi_cls = inst_infos["category_id"]

        # extent
        roi_extent = self._get_extents(dataset_name)[roi_cls]
        dataset_dict["roi_extent"] = torch.tensor(roi_extent, dtype=torch.float32)

        # load xyz =======================================================
        xyz_info = mmcv.load(inst_infos["xyz_path"])
        x1, y1, x2, y2 = xyz_info["xyxy"]
        # float16 does not affect performance (classification/regresion)
        xyz_crop = xyz_info["xyz_crop"]
        xyz = np.zeros((im_H, im_W, 3), dtype=np.float32)
        xyz[y1 : y2 + 1, x1 : x2 + 1, :] = xyz_crop
        # NOTE: full mask
        mask_obj = ((xyz[:, :, 0] != 0) | (xyz[:, :, 1] != 0) | (xyz[:, :, 2] != 0)).astype(np.bool).astype(np.float32)
        if cfg.input.smooth_xyz:
            xyz = self.smooth_xyz(xyz)

        if cfg.train.vis:
            xyz = self.smooth_xyz(xyz)

        if cfg.model.bbox_crop_syn and "syn" in img_type:
            inst_infos["bbox"] = inst_infos["bbox_crop"]
        elif cfg.model.bbox_crop_real and "real" in img_type:
            inst_infos["bbox"] = inst_infos["bbox_crop"]
        else:
            # override bbox info using xyz_infos
            inst_infos["bbox"] = [x1, y1, x2, y2]
            inst_infos["bbox_mode"] = BoxMode.XYXY_ABS

        # USER: Implement additional transformations if you have other types of data
        # inst_infos.pop("segmentation")  # NOTE: use mask from xyz
        anno = transform_instance_annotations(inst_infos, transforms, image_shape, keypoint_hflip_indices=None)

        # augment bbox ===================================================
        bbox_xyxy = anno["bbox"]
        bbox_center, scale = self.aug_bbox_DZI(cfg, bbox_xyxy, im_H, im_W)
        bw = max(bbox_xyxy[2] - bbox_xyxy[0], 1)
        bh = max(bbox_xyxy[3] - bbox_xyxy[1], 1)

        # CHW, float32 tensor
        ## roi_image ------------------------------------
        roi_img = crop_resize_by_warp_affine(
            image,
            bbox_center,
            scale,
            input_res,
            interpolation=cv2.INTER_LINEAR,
        ).transpose(2, 0, 1)

        roi_img = self.normalize_image(cfg, roi_img)

        # roi_coord_2d ----------------------------------------------------
        roi_coord_2d = crop_resize_by_warp_affine(
            coord_2d,
            bbox_center,
            scale,
            out_res,
            interpolation=cv2.INTER_LINEAR,
        ).transpose(2, 0, 1)

        # roi_coord_2d_rel
        roi_coord_2d_rel = (
            bbox_center.reshape(2, 1, 1) - roi_coord_2d * np.array([im_W, im_H]).reshape(2, 1, 1)
        ) / scale

        ## roi_mask ---------------------------------------
        # (mask_trunc < mask_visib < mask_obj)
        mask_visib = anno["segmentation"].astype("float32") * mask_obj

        if mask_trunc is None:
            mask_trunc = mask_visib
        else:
            mask_trunc = mask_visib * mask_trunc.astype("float32")

        # maybe truncated mask (true mask for rgb)
        roi_mask_trunc = crop_resize_by_warp_affine(
            mask_trunc[:, :, None],
            bbox_center,
            scale,
            out_res,
            interpolation=mask_xyz_interp,
        )

        # use original visible mask to calculate xyz loss (try full obj mask?)
        roi_mask_visib = crop_resize_by_warp_affine(
            mask_visib[:, :, None],
            bbox_center,
            scale,
            out_res,
            interpolation=mask_xyz_interp,
        )

        roi_mask_obj = crop_resize_by_warp_affine(
            mask_obj[:, :, None],
            bbox_center,
            scale,
            out_res,
            interpolation=mask_xyz_interp,
        )

        if "mask_full" in anno.keys():
            # TODO: maybe directly use mask_obj
            mask_full = anno["mask_full"].astype("float32")
            roi_mask_full = crop_resize_by_warp_affine(
                mask_full[:, :, None],
                bbox_center,
                scale,
                out_res,
                interpolation=mask_xyz_interp,
            )

        ## roi_xyz ----------------------------------------------------
        roi_xyz = crop_resize_by_warp_affine(xyz, bbox_center, scale, out_res, interpolation=mask_xyz_interp)

        # region label
        if g_head_cfg.num_regions > 1:
            fps_points = self._get_fps_points(dataset_name)[roi_cls]
            roi_region = xyz_to_region(roi_xyz, fps_points)  # HW
            dataset_dict["roi_region"] = torch.as_tensor(roi_region.astype(np.int32)).contiguous()

        roi_xyz = roi_xyz.transpose(2, 0, 1)  # HWC-->CHW
        # normalize xyz to [0, 1] using extent
        roi_xyz[0] = roi_xyz[0] / roi_extent[0] + 0.5
        roi_xyz[1] = roi_xyz[1] / roi_extent[1] + 0.5
        roi_xyz[2] = roi_xyz[2] / roi_extent[2] + 0.5

        xyz_loss_type = loss_cfg.xyz_loss_type
        if ("CE" in xyz_loss_type) or ("cls" in cfg.model.pose_net.name):  # convert target to int for cls
            n_xyz_bin = g_head_cfg.xyz_bin
            # assume roi_xyz has been normalized in [0, 1]
            roi_xyz_bin = np.zeros_like(roi_xyz)
            roi_x_norm = roi_xyz[0]
            roi_x_norm[roi_x_norm < 0] = 0  # clip
            roi_x_norm[roi_x_norm > 0.999999] = 0.999999
            # [0, BIN-1]
            roi_xyz_bin[0] = np.asarray(roi_x_norm * n_xyz_bin, dtype=np.uint8)

            roi_y_norm = roi_xyz[1]
            roi_y_norm[roi_y_norm < 0] = 0
            roi_y_norm[roi_y_norm > 0.999999] = 0.999999
            roi_xyz_bin[1] = np.asarray(roi_y_norm * n_xyz_bin, dtype=np.uint8)

            roi_z_norm = roi_xyz[2]
            roi_z_norm[roi_z_norm < 0] = 0
            roi_z_norm[roi_z_norm > 0.999999] = 0.999999
            roi_xyz_bin[2] = np.asarray(roi_z_norm * n_xyz_bin, dtype=np.uint8)

            # the last bin is for bg
            roi_masks = {
                "trunc": roi_mask_trunc,
                "visib": roi_mask_visib,
                "obj": roi_mask_obj,
                "full": roi_mask_full,
            }
            roi_mask_xyz = roi_masks[loss_cfg.xyz_loss_mask_gt]
            roi_xyz_bin[0][roi_mask_xyz == 0] = n_xyz_bin
            roi_xyz_bin[1][roi_mask_xyz == 0] = n_xyz_bin
            roi_xyz_bin[2][roi_mask_xyz == 0] = n_xyz_bin

            if "CE" in xyz_loss_type:
                dataset_dict["roi_xyz_bin"] = torch.as_tensor(roi_xyz_bin.astype("uint8")).contiguous()
            if "/" in xyz_loss_type and len(xyz_loss_type.split("/")[1]) > 0:
                dataset_dict["roi_xyz"] = torch.as_tensor(roi_xyz.astype("float32")).contiguous()
        else:
            dataset_dict["roi_xyz"] = torch.as_tensor(roi_xyz.astype("float32")).contiguous()

        # pose targets ----------------------------------------------------------------------
        pose = inst_infos["pose"]
        dataset_dict["ego_rot"] = torch.as_tensor(pose[:3, :3].astype("float32"))
        dataset_dict["trans"] = torch.as_tensor(inst_infos["trans"].astype("float32"))

        dataset_dict["roi_points"] = torch.as_tensor(self._get_model_points(dataset_name)[roi_cls].astype("float32"))
        dataset_dict["sym_info"] = self._get_sym_infos(dataset_name)[roi_cls]

        dataset_dict["roi_img"] = torch.as_tensor(roi_img.astype("float32")).contiguous()
        dataset_dict["roi_coord_2d"] = torch.as_tensor(roi_coord_2d.astype("float32")).contiguous()
        dataset_dict["roi_coord_2d_rel"] = torch.as_tensor(roi_coord_2d_rel.astype("float32")).contiguous()

        dataset_dict["roi_mask_trunc"] = torch.as_tensor(roi_mask_trunc.astype("float32")).contiguous()
        dataset_dict["roi_mask_visib"] = torch.as_tensor(roi_mask_visib.astype("float32")).contiguous()
        dataset_dict["roi_mask_obj"] = torch.as_tensor(roi_mask_obj.astype("float32")).contiguous()
        if "mask_full" in anno.keys():
            dataset_dict["roi_mask_full"] = torch.as_tensor(roi_mask_full.astype("float32")).contiguous()

        dataset_dict["bbox_center"] = torch.as_tensor(bbox_center, dtype=torch.float32)
        dataset_dict["scale"] = scale
        dataset_dict["bbox"] = anno["bbox"]  # NOTE: original bbox
        dataset_dict["roi_wh"] = torch.as_tensor(np.array([bw, bh], dtype=np.float32))
        dataset_dict["resize_ratio"] = resize_ratio = out_res / scale
        z_ratio = inst_infos["trans"][2] / resize_ratio
        obj_center = anno["centroid_2d"]
        delta_c = obj_center - bbox_center
        dataset_dict["trans_ratio"] = torch.as_tensor([delta_c[0] / bw, delta_c[1] / bh, z_ratio]).to(torch.float32)
        return dataset_dict

    def smooth_xyz(self, xyz):
        """smooth the edge areas to reduce noise."""
        xyz = np.asarray(xyz, np.float32)
        xyz_blur = cv2.medianBlur(xyz, 3)
        edges = get_edge(xyz)
        xyz[edges != 0] = xyz_blur[edges != 0]
        return xyz

    def __getitem__(self, idx):
        if self.split != "train":
            dataset_dict = self._get_sample_dict(idx)
            return self.read_data(dataset_dict)

        while True:  # return valid data for train
            dataset_dict = self._get_sample_dict(idx)
            processed_data = self.read_data(dataset_dict)
            if processed_data is None:
                idx = self._rand_another(idx)
                continue
            return processed_data

class GDRN_Self_DatasetFromList(Base_DatasetFromList):
    """NOTE: we can also use the default DatasetFromList and implement a similar custom DataMapper,
    but it is harder to implement some features relying on other dataset dicts.
    # https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/common.py
    Wrap a list to a torch Dataset. It produces elements of the list as data.
    """

    def __init__(self, cfg, split, lst: list, copy: bool = True, serialize: bool = True, flatten=True):
        """
        Args:
            lst (list): a list which contains elements to produce.
            copy (bool): whether to deepcopy the element when producing it,
                so that the result can be modified in place without affecting the
                source in the list.
            serialize (bool): whether to hold memory using serialized objects, when
                enabled, data loader workers can use shared RAM from master
                process instead of making a copy.
        """
        self.augmentation = build_gdrn_augmentation(cfg, is_train=(split == "train"))
        # fmt: off
        self.img_format = cfg.input.format  # default bgr
        self.with_depth = cfg.input.with_depth
        self.aug_depth = cfg.input.aug_depth
        # note: color augmentation config
        self.color_aug_prob = cfg.input.color_aug_prob
        self.color_aug_type = cfg.input.color_aug_type
        self.color_aug_code = cfg.input.color_aug_code
        # fmt: on
        self.cfg = cfg
        self.split = split  # train | val | test
        if split == "train" and self.color_aug_prob > 0:
            self.color_augmentor = self._get_color_augmentor(aug_type=self.color_aug_type, aug_code=self.color_aug_code)
        else:
            self.color_augmentor = None
        # ------------------------
        # common model infos
        self.fps_points = {}
        self.model_points = {}
        self.extents = {}
        self.sym_infos = {}
        # ----------------------------------------------------
        self.flatten = flatten
        self._lst = flat_dataset_dicts(lst) if flatten else lst
        # ----------------------------------------------------
        self._copy = copy
        self._serialize = serialize

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        if self._serialize:
            logger.info("Serializing {} elements to byte tensors and concatenating them all ...".format(len(self._lst)))
            self._lst = [_serialize(x) for x in self._lst]
            self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
            self._addr = np.cumsum(self._addr)
            self._lst = np.concatenate(self._lst)
            logger.info("Serialized dataset takes {:.2f} MiB".format(len(self._lst) / 1024 ** 2))

    def __len__(self):
        if self._serialize:
            return len(self._addr)
        else:
            return len(self._lst)

    def _get_fps_points(self, dataset_name, with_center=False):
        """convert to label based keys.

        # TODO: get models info similarly
        """
        if dataset_name in self.fps_points:
            return self.fps_points[dataset_name]

        dset_meta = MetadataCatalog.get(dataset_name)
        ref_key = dset_meta.ref_key
        data_ref = ref.__dict__[ref_key]
        objs = dset_meta.objs
        cfg = self.cfg
        num_fps_points = cfg.model.pose_net.geo_head.num_regions
        cur_fps_points = {}
        loaded_fps_points = data_ref.get_fps_points()
        for i, obj_name in enumerate(objs):
            obj_id = data_ref.obj2id[obj_name]
            if with_center:
                cur_fps_points[i] = loaded_fps_points[str(obj_id)][f"fps{num_fps_points}_and_center"]
            else:
                cur_fps_points[i] = loaded_fps_points[str(obj_id)][f"fps{num_fps_points}_and_center"][:-1]
        self.fps_points[dataset_name] = cur_fps_points
        return self.fps_points[dataset_name]

    def _get_model_points(self, dataset_name):
        """convert to label based keys."""
        if dataset_name in self.model_points:
            return self.model_points[dataset_name]

        dset_meta = MetadataCatalog.get(dataset_name)
        ref_key = dset_meta.ref_key
        data_ref = ref.__dict__[ref_key]
        objs = dset_meta.objs
        cfg = self.cfg

        cur_model_points = {}
        num = np.inf
        for i, obj_name in enumerate(objs):
            obj_id = data_ref.obj2id[obj_name]
            model_path = osp.join(data_ref.model_dir, f"obj_{obj_id:06d}.ply")
            model = inout.load_ply(model_path, vertex_scale=data_ref.vertex_scale)
            cur_model_points[i] = pts = model["pts"]
            if pts.shape[0] < num:
                num = pts.shape[0]

        num = min(num, cfg.train.loss.num_pm_points)
        for i in range(len(cur_model_points)):
            keep_idx = np.arange(num)
            np.random.shuffle(keep_idx)  # random sampling
            cur_model_points[i] = cur_model_points[i][keep_idx, :]

        self.model_points[dataset_name] = cur_model_points
        return self.model_points[dataset_name]

    def _get_extents(self, dataset_name):
        """label based keys."""
        if dataset_name in self.extents:
            return self.extents[dataset_name]

        dset_meta = MetadataCatalog.get(dataset_name)
        try:
            ref_key = dset_meta.ref_key
        except:
            # FIXME: for some reason, in distributed training, this need to be re-registered
            register_datasets([dataset_name])
            dset_meta = MetadataCatalog.get(dataset_name)
            ref_key = dset_meta.ref_key

        data_ref = ref.__dict__[ref_key]
        objs = dset_meta.objs

        cur_extents = {}
        for i, obj_name in enumerate(objs):
            obj_id = data_ref.obj2id[obj_name]
            model_path = osp.join(data_ref.model_dir, f"obj_{obj_id:06d}.ply")
            model = inout.load_ply(model_path, vertex_scale=data_ref.vertex_scale)
            pts = model["pts"]
            xmin, xmax = np.amin(pts[:, 0]), np.amax(pts[:, 0])
            ymin, ymax = np.amin(pts[:, 1]), np.amax(pts[:, 1])
            zmin, zmax = np.amin(pts[:, 2]), np.amax(pts[:, 2])
            size_x = xmax - xmin
            size_y = ymax - ymin
            size_z = zmax - zmin
            cur_extents[i] = np.array([size_x, size_y, size_z], dtype="float32")

        self.extents[dataset_name] = cur_extents
        return self.extents[dataset_name]

    def _get_sym_infos(self, dataset_name):
        """label based keys."""
        if dataset_name in self.sym_infos:
            return self.sym_infos[dataset_name]

        dset_meta = MetadataCatalog.get(dataset_name)
        ref_key = dset_meta.ref_key
        data_ref = ref.__dict__[ref_key]
        objs = dset_meta.objs

        cur_sym_infos = {}
        loaded_models_info = data_ref.get_models_info()
        for i, obj_name in enumerate(objs):
            obj_id = data_ref.obj2id[obj_name]
            model_info = loaded_models_info[str(obj_id)]
            if "symmetries_discrete" in model_info or "symmetries_continuous" in model_info:
                sym_transforms = misc.get_symmetry_transformations(model_info, max_sym_disc_step=0.01)
                sym_info = np.array([sym["R"] for sym in sym_transforms], dtype=np.float32)
            else:
                sym_info = None
            cur_sym_infos[i] = sym_info

        self.sym_infos[dataset_name] = cur_sym_infos
        return self.sym_infos[dataset_name]

    def read_data(self, dataset_dict, idx):
        """load image and annos; random shift & scale bbox; crop, rescale."""
        cfg = self.cfg
        net_cfg = cfg.model.pose_net
        g_head_cfg = net_cfg.geo_head

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        dataset_name = dataset_dict["dataset_name"]

        image = read_image_mmcv(dataset_dict["file_name"], format=self.img_format)
        pseudo_syn_image = read_image_mmcv(dataset_dict["rgb_syn_file"], format=self.img_format)
        pseudo_mask = read_image_mmcv(dataset_dict["mask_syn_file"], format="unchanged")
        # print(dataset_dict["rgb_syn_file"])
        # should be consistent with the size in dataset_dict
        utils.check_image_size(dataset_dict, image) 
        im_H_ori, im_W_ori = image.shape[:2]

        # NOTE: assume loading real images with detections, so no replacing bg ###############################

        gt_img = image.copy()  # original image, no color aug

        # NOTE: maybe add or change color augment here ===================================
        if self.split == "train" and self.color_aug_prob > 0 and self.color_augmentor is not None:
            if np.random.rand() < self.color_aug_prob:
                image = self._color_aug(image, self.color_aug_type)

        # other transforms (mainly geometric ones);
        # for 6d pose task, flip is not allowed in general except for some 2d keypoints methods
        image, transforms = T.apply_augmentations(self.augmentation, image)
        # gt_img after geometric aug (maybe no geo aug)
        gt_img, _ = T.apply_augmentations(self.augmentation, gt_img)
        im_H, im_W = image_shape = image.shape[:2]  # h, w

        # NOTE: scale camera intrinsic if necessary ================================
        scale_x = im_W / im_W_ori
        scale_y = im_H / im_H_ori  # NOTE: generally scale_x should be equal to scale_y
        if "cam" in dataset_dict:
            if im_W != im_W_ori or im_H != im_H_ori:
                dataset_dict["cam"][0] *= scale_x
                dataset_dict["cam"][1] *= scale_y
            K = dataset_dict["cam"].astype("float32")
            dataset_dict["cam"] = torch.as_tensor(K)

        input_res = net_cfg.input_res
        out_res = net_cfg.output_res

        # CHW -> HWC
        coord_2d = get_2d_coord_np(im_W, im_H, low=0, high=1).transpose(1, 2, 0)

        #################################################################################
        if self.split != "train":
            # don't load annotations at test time
            test_bbox_type = cfg.test.test_bbox_type
            if test_bbox_type == "gt":
                bbox_key = "bbox"
            else:
                bbox_key = f"bbox_{test_bbox_type}"
            assert not self.flatten, "Do not use flattened dicts for test!"
            # here get batched rois
            roi_infos = {}
            # yapf: disable
            roi_keys = ["scene_im_id", "file_name", "cam", "im_H", "im_W",
                        "roi_img", "inst_id", "roi_coord_2d", "roi_cls", "score", "roi_extent",
                         bbox_key, "bbox_mode", "bbox_center", "roi_wh",
                         "scale", "resize_ratio", "model_info",
                        ]
            for _key in roi_keys:
                roi_infos[_key] = []
            # yapf: enable
            # TODO: how to handle image without detections
            #   filter those when load annotations or detections, implement a function for this
            # "annotations" means detections
            for inst_i, inst_infos in enumerate(dataset_dict["annotations"]):
                # inherent image-level infos
                roi_infos["scene_im_id"].append(dataset_dict["scene_im_id"])
                roi_infos["file_name"].append(dataset_dict["file_name"])
                roi_infos["im_H"].append(im_H)
                roi_infos["im_W"].append(im_W)
                roi_infos["cam"].append(dataset_dict["cam"].cpu().numpy())

                # roi-level infos
                roi_infos["inst_id"].append(inst_i)
                roi_infos["model_info"].append(inst_infos["model_info"])

                roi_cls = inst_infos["category_id"]
                roi_infos["roi_cls"].append(roi_cls)
                roi_infos["score"].append(inst_infos.get("score", 1.0))

                # extent
                roi_extent = self._get_extents(dataset_name)[roi_cls]
                roi_infos["roi_extent"].append(roi_extent)

                bbox = BoxMode.convert(
                    inst_infos[bbox_key],
                    inst_infos["bbox_mode"],
                    BoxMode.XYXY_ABS,
                )
                bbox = np.array(transforms.apply_box([bbox])[0])
                roi_infos[bbox_key].append(bbox)
                roi_infos["bbox_mode"].append(BoxMode.XYXY_ABS)
                x1, y1, x2, y2 = bbox
                bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
                bw = max(x2 - x1, 1)
                bh = max(y2 - y1, 1)
                scale = max(bh, bw) * cfg.input.dzi_pad_scale
                scale = min(scale, max(im_H, im_W)) * 1.0

                roi_infos["bbox_center"].append(bbox_center.astype("float32"))
                roi_infos["scale"].append(scale)
                roi_infos["roi_wh"].append(np.array([bw, bh], dtype=np.float32))
                roi_infos["resize_ratio"].append(out_res / scale)

                # CHW, float32 tensor
                # roi_image
                roi_img = crop_resize_by_warp_affine(
                    image,
                    bbox_center,
                    scale,
                    input_res,
                    interpolation=cv2.INTER_LINEAR,
                ).transpose(2, 0, 1)

                roi_img = self.normalize_image(cfg, roi_img)
                roi_infos["roi_img"].append(roi_img.astype("float32"))

                # roi_coord_2d
                roi_coord_2d = crop_resize_by_warp_affine(
                    coord_2d,
                    bbox_center,
                    scale,
                    out_res,
                    interpolation=cv2.INTER_LINEAR,
                ).transpose(
                    2, 0, 1
                )  # HWC -> CHW
                roi_infos["roi_coord_2d"].append(roi_coord_2d.astype("float32"))

            for _key in roi_keys:
                if _key in ["roi_img", "roi_coord_2d"]:
                    dataset_dict[_key] = torch.as_tensor(roi_infos[_key]).contiguous()
                elif _key in ["model_info", "scene_im_id", "file_name"]:
                    # can not convert to tensor
                    dataset_dict[_key] = roi_infos[_key]
                else:
                    dataset_dict[_key] = torch.tensor(roi_infos[_key])

            return dataset_dict
        #######################################################################################
        # NOTE: currently assume flattened dicts for train
        assert self.flatten, "Only support flattened dicts for train now"
        inst_infos = dataset_dict.pop("inst_infos")
        dataset_dict["roi_cls"] = roi_cls = inst_infos["category_id"]

        # extent
        roi_extent = self._get_extents(dataset_name)[roi_cls]
        dataset_dict["roi_extent"] = torch.tensor(roi_extent, dtype=torch.float32)

        img_type = dataset_dict.get("img_type", "real")
        if cfg.model.load_dets_train:
            bbox_key = "bbox_est"
            
        # USER: Implement additional transformations if you have other types of data
        anno = transform_instance_annotations(
            inst_infos, transforms, image_shape, keypoint_hflip_indices=None, bbox_key=bbox_key
        )

        # augment bbox ===================================================
        bbox_xyxy = anno[bbox_key]
        bbox_center, scale = self.aug_bbox_DZI(cfg, bbox_xyxy, im_H, im_W)
        bw = max(bbox_xyxy[2] - bbox_xyxy[0], 1)
        bh = max(bbox_xyxy[3] - bbox_xyxy[1], 1)

        ## load pseudo pose if configured ----------------------------
        if cfg.model.load_dets_train_with_pose:
            pose_est = np.array(anno["pose_est"], dtype=np.float32)
            pose_refine = np.array(anno["pose_refine"], dtype=np.float32)
            dataset_dict["pose_est"] = torch.as_tensor(pose_est)
            dataset_dict["pose_refine"] = torch.as_tensor(pose_refine)

        # CHW, float32 tensor
        ## roi_image ------------------------------------
        roi_img = crop_resize_by_warp_affine(
            image, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR
        ).transpose(2, 0, 1)

        roi_img = self.normalize_image(cfg, roi_img)

        ## roi_image ori
        roi_gt_img = crop_resize_by_warp_affine(
            gt_img, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR
        ).transpose(2, 0, 1)
        roi_gt_img = self.normalize_image(cfg, roi_gt_img)

        # roi_syn_image
        roi_syn_image = crop_resize_by_warp_affine(
            pseudo_syn_image, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR
        ).transpose(2, 0, 1)
        roi_syn_image = self.normalize_image(cfg, roi_syn_image)
        # roi_syn_mask
        roi_syn_mask = crop_resize_by_warp_affine(pseudo_mask, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR)
        roi_syn_mask =  self.normalize_mask(cfg, roi_syn_mask)
        ## load depth
        if self.with_depth:
            assert "depth_file" in dataset_dict, "depth file is not in dataset_dict"
            depth_path = dataset_dict["depth_file"]
            depth = mmcv.imread(depth_path, "unchanged") / dataset_dict["depth_factor"]  # to m

            if self.aug_depth:  # NOTE: augment depth
                depth[depth == 0] = np.random.normal(np.median(depth[depth == 0]), 0.1, depth[depth == 0].shape)
                if np.random.randint(2) < 1:  # drop 20% of depth values
                    mask = np.random.random_integers(0, 10, size=depth.shape)
                    mask = mask > 1
                    depth = depth * mask

                depth_idx = depth > 0
                depth[depth_idx] += np.random.normal(0, 0.01, depth[depth_idx].shape)
            dataset_dict["depth"] = torch.as_tensor(depth.reshape(im_H, im_W).astype("float32"))

        # roi_coord_2d ----------------------------------------------------
        roi_coord_2d = crop_resize_by_warp_affine(
            coord_2d, bbox_center, scale, out_res, interpolation=cv2.INTER_LINEAR
        ).transpose(2, 0, 1)

        # fps points: for region label
        if g_head_cfg.num_regions > 1:
            fps_points = self._get_fps_points(dataset_name)[roi_cls]
            dataset_dict["roi_fps_points"] = torch.as_tensor(fps_points.astype(np.float32)).contiguous()

        dataset_dict["roi_img"] = torch.as_tensor(roi_img.astype("float32")).contiguous()
        dataset_dict["roi_gt_img"] = torch.as_tensor(roi_gt_img.astype("float32")).contiguous()
        
        pseudo_syn_image = self.normalize_image(cfg, pseudo_syn_image.transpose(2, 0, 1))
        dataset_dict['syn_image'] = torch.as_tensor(pseudo_syn_image.astype("float32")).contiguous()
        dataset_dict['roi_syn_image'] = torch.as_tensor(roi_syn_image.astype("float32")).contiguous()
        pseudo_mask = self.normalize_mask(cfg, pseudo_mask)
        dataset_dict['syn_mask'] = torch.as_tensor(pseudo_mask.astype("float32")).contiguous()
        dataset_dict['roi_syn_mask'] = torch.as_tensor(roi_syn_mask.astype("float32")).contiguous()
        
        gt_img = self.normalize_image(cfg, gt_img.transpose(2, 0, 1))
        dataset_dict["gt_img"] = torch.as_tensor(gt_img.astype("float32")).contiguous()

        dataset_dict["roi_points"] = torch.as_tensor(self._get_model_points(dataset_name)[roi_cls].astype("float32"))
        dataset_dict["sym_info"] = self._get_sym_infos(dataset_name)[roi_cls]

        dataset_dict["roi_coord_2d"] = torch.as_tensor(roi_coord_2d.astype("float32")).contiguous()

        dataset_dict["bbox_center"] = torch.as_tensor(bbox_center, dtype=torch.float32)
        dataset_dict["scale"] = scale
        dataset_dict["bbox"] = anno[bbox_key]  # NOTE: original bbox
        dataset_dict["roi_wh"] = torch.as_tensor(np.array([bw, bh], dtype=np.float32))
        dataset_dict["resize_ratio"] = resize_ratio = out_res / scale

        return dataset_dict

    def __getitem__(self, idx):
        if self.split != "train":
            dataset_dict = self._get_sample_dict(idx)
            return self.read_data(dataset_dict, idx)

        while True:  # return valid data for train
            dataset_dict = self._get_sample_dict(idx)
            processed_data = self.read_data(dataset_dict, idx)
            if processed_data is None:
                idx = self._rand_another(idx)
                continue
            return processed_data


def load_detections_with_poses_into_dataset(
    dataset_name,
    dataset_dicts,
    det_file,
    top_k_per_obj=1,
    score_thr=0.0,
    train_objs=None,
    top_k_per_im=None,
    with_pose=False,
):
    """Load test detections into the dataset.

    Args:
        dataset_name (str):
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.
        det_file (str): file path of pre-computed detections, in json format.

    Returns:
        list[dict]: the same format as dataset_dicts, but added proposal field.
    """

    logger.info("Loading detections for {} from: {}".format(dataset_name, det_file))
    detections = mmcv.load(det_file)

    meta = MetadataCatalog.get(dataset_name)
    objs = meta.objs
    ref_key = meta.ref_key
    data_ref = ref.__dict__[ref_key]
    models_info = data_ref.get_models_info()

    if "annotations" in dataset_dicts[0]:
        logger.warning("pop the original annotations, load detections")
    new_dataset_dicts = []
    for i, record_ori in enumerate(dataset_dicts):
        record = copy.deepcopy(record_ori)
        scene_im_id = record["scene_im_id"]
        if scene_im_id not in detections:  # not detected
            logger.warning(f"no detections found in {scene_im_id}")
            continue
        dets_i = detections[scene_im_id]

        annotations = []
        obj_annotations = {obj: [] for obj in objs}
        for det in dets_i:
            score = det.get("score", 1.0)
            if score < score_thr:
                continue

            obj_id = det["obj_id"]
            obj_name = data_ref.id2obj[obj_id]
            if obj_name not in objs:  # detected obj is not interested
                continue

            if train_objs is not None:  # not in trained objects
                if obj_name not in train_objs:
                    continue

            bbox_est = det["bbox_est"]  # xywh
            time = det.get("time", 0.0)

            label = objs.index(obj_name)
            inst = {
                "category_id": label,
                "bbox_est": bbox_est,
                "bbox_mode": BoxMode.XYWH_ABS,
                "score": score,
                "time": time,
                "model_info": models_info[str(obj_id)],  # TODO: maybe just load this in the main function
            }
            if with_pose:
                log_first_n(logging.INFO, "load detections with pose_est and pose_refine", n=1)
                assert "pose_est" in det and "pose_refine" in det
                inst["pose_est"] = det["pose_est"]
                inst["pose_refine"] = det["pose_refine"]
            obj_annotations[obj_name].append(inst)
        for obj, cur_annos in obj_annotations.items():
            scores = [ann["score"] for ann in cur_annos]
            sel_annos = [ann for _, ann in sorted(zip(scores, cur_annos), key=lambda pair: pair[0], reverse=True)][
                :top_k_per_obj
            ]
            annotations.extend(sel_annos)
        # NOTE: maybe [], no detections
        record["annotations"] = annotations
        new_dataset_dicts.append(record)

    if len(new_dataset_dicts) < len(dataset_dicts):
        logger.warning(
            "No detections found in {} images. original: {} imgs, left: {} imgs".format(
                len(dataset_dicts) - len(new_dataset_dicts), len(dataset_dicts), len(new_dataset_dicts)
            )
        )
    return new_dataset_dicts

import sys
def build_gdrn_self_train_loader(cfg, dataset_names, train_objs=None):
    """A data loader is created by the following steps:

    1. Use the dataset names in config to query :class:`DatasetCatalog`, and obtain a list of dicts.
    2. Coordinate a random shuffle order shared among all processes (all GPUs)
    3. Each process spawn another few workers to process the dicts. Each worker will:
       * Map each metadata dict into another format to be consumed by the model.
       * Batch them by simply putting dicts into a list.

    The batched ``list[mapped_dict]`` is what this dataloader will yield.

    Args:
        cfg (CfgNode): the config

    Returns:
        an infinite iterator of training data
    """
    dataset_dicts_list = [
        get_detection_dataset_dicts(
            [dataset_name],
            filter_empty=cfg.dataloader.filter_empty_annotations,
            min_keypoints=cfg.model.roi_keypoint_head.min_keypoints_per_image if cfg.model.keypoint_on else 0,
            proposal_files=None,
        )
        for dataset_name in dataset_names
    ]
    # print("dataset_dicts_list: ", dataset_dicts_list)
    # with open('crack_box.txt', 'w') as f:
    #     for scene in dataset_dicts_list:
    #         for data in scene:
    #             # print(data)
    #             # sys.exit()
    #             scene_id = "{:04d}".format(int(data['file_name'].split('/')[4]))
    #             img_id = (data['file_name'].split('/')[-1]).split('.')[0]
    #             f.write("{}/{}".format(scene_id, img_id))
    #             f.write('\n')
    # sys.exit()
    # load detection results for training sets for self-supervised training
    if cfg.model.self_train and cfg.model.load_dets_train:
        dataset_dicts_list_new = []
        det_files = cfg.datasets.det_files_train
        assert len(cfg.datasets.train) == len(det_files)
        for dataset_name, dataset_dicts, det_file in zip(dataset_names, dataset_dicts_list, det_files):
            dataset_dicts = load_detections_with_poses_into_dataset(
                dataset_name,
                dataset_dicts,
                det_file=det_file,
                top_k_per_obj=cfg.datasets.det_topk_per_obj_train,
                score_thr=cfg.datasets.det_thr_train,
                train_objs=train_objs,
                with_pose=cfg.model.load_dets_train_with_pose,
            )
            if cfg.dataloader.filter_empty_dets:
                dataset_dicts = filter_empty_dets(dataset_dicts)
            dataset_dicts_list_new.append(dataset_dicts)
    else:
        dataset_dicts_list_new = dataset_dicts_list

    final_dataset_dicts = []
    for dataset_dicts in dataset_dicts_list_new:
        final_dataset_dicts.extend(dataset_dicts)

    final_dataset_dicts = filter_invalid_in_dataset_dicts(
        final_dataset_dicts, visib_thr=cfg.dataloader.filter_visib_thr
    )

    dataset = GDRN_Self_DatasetFromList(cfg, split="train", lst=final_dataset_dicts, copy=False, flatten=True)

    sampler_name = cfg.dataloader.sampler_train
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    # TODO avoid if-else?
    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == "TrainingSampler_Pair":
        path_data = cfg.datasets.camera_pose_path
        list_scene_num = []
        interval = int(cfg.dataloader.interval)
        with open(path_data + "image_scene_num.txt", "r") as f:
            list_scene_num = list(map(int, f.readlines()))
        sampler = TrainingSampler_Pair(len(dataset), list_scene_num, interval)
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            final_dataset_dicts, cfg.dataloader.repeat_threshold
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))
    return my_build_batch_data_loader(
        dataset,
        sampler,
        cfg.train.batch_size,
        aspect_ratio_grouping=cfg.dataloader.aspect_ratio_grouping,
        num_workers=cfg.dataloader.num_workers,
    )

def build_gdrn_test_loader(cfg, dataset_name, train_objs=None):
    """Similar to `build_detection_train_loader`. But this function uses the
    given `dataset_name` argument (instead of the names in cfg), and uses batch
    size 1.

    Args:
        cfg: a detectron2 CfgNode
        dataset_name (str): a name of the dataset that's available in the DatasetCatalog

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.
    """
    dataset_dicts = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
        proposal_files=None,
    )

    # load test detection results
    if cfg.model.load_dets_test:
        det_files = cfg.datasets.det_files_test
        assert len(cfg.datasets.test) == len(det_files)
        dataset_dicts = load_detections_into_dataset(
            dataset_name,
            dataset_dicts,
            det_file=det_files[cfg.datasets.test.index(dataset_name)],
            top_k_per_obj=cfg.datasets.det_topk_per_obj,
            score_thr=cfg.datasets.det_thr,
            train_objs=train_objs,
        )
        if cfg.dataloader.filter_empty_dets:
            dataset_dicts = filter_empty_dets(dataset_dicts)

    dataset = GDRN_DatasetFromList(cfg, split="test", lst=dataset_dicts, flatten=False)

    sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    num_workers = cfg.dataloader.num_workers
    # Horovod: limit # of CPU threads to be used per worker.
    # if num_workers > 0:
    #     torch.set_num_threads(num_workers)

    kwargs = {"num_workers": num_workers}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    # https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_imagenet_resnet50.py
    # if (num_workers > 0 and hasattr(mp, '_supports_context') and
    #         mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
    #     kwargs['multiprocessing_context'] = 'forkserver'
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
        **kwargs,
    )
    return data_loader
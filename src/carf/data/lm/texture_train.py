import json
import os
import PIL
import imageio
import numpy as np
import torch
import cv2
import logging
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import time
from src.carf.utils import camera 
from .. import base
from omegaconf import OmegaConf


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

logger = logging.getLogger(__name__)

class Dataset(base.Dataset):
    def __init__(self, cfg, split="train", subset=None, multi_obj=False):
        super().__init__(cfg, split)
        self.data_path = os.path.join(cfg.data.root, cfg.data.dataset) 

        self.split_path = os.path.join("splits", cfg.data.dataset, str(cfg.data.object),
                                       cfg.data.scene, "{}.txt".format(split))
        with open(self.split_path, "r") as f:
            self.list = f.readlines()
        self.list = [line.strip() for line in self.list]
        self.multi_obj = multi_obj

        self.initialize_meta(cfg)
    
    
    def setup_loader(self, cfg, shuffle=False, drop_last=False):
        loader = torch.utils.data.DataLoader(self,
                                             batch_size=cfg.batch_size_gan or 1,
                                             num_workers=cfg.data.num_workers,
                                             shuffle=shuffle,
                                             drop_last=drop_last,
                                             pin_memory=False)
        print("number of samples: {}".format(len(self)))
        return loader

    def initialize_meta(self, cfg):
        line = self.list[0].split(' ')
        model_name, folder = line[0], line[1]
        scene_obj_path = os.path.join(self.data_path, folder, 'scene_object.json')

        # load scene info
        if self.split != 'test' and cfg.data.pose_source == 'predicted':
            if cfg.data.scene_info_source is None:
                scene_info_path = os.path.join(self.data_path, folder, 'scene_pred_info.json')
            else:
                info_names = dict(gt='scene_gt_info.json', predicted='scene_pred_info.json')
                scene_info_path = os.path.join(self.data_path, folder, info_names[cfg.data.scene_info_source])
        else:
            scene_info_path = os.path.join(self.data_path, folder, 'scene_gt_info.json')

        with open(scene_info_path) as f:
            self.scene_info_all = json.load(f)

        # load scene pose
        scene_cam_path = os.path.join(self.data_path, folder, 'scene_camera.json')
        scene_pred_path = os.path.join(self.data_path, folder, 'scene_pred.json')
        logger.info("Use predicted pose from {} ...".format(scene_pred_path))

        with open(scene_cam_path) as f:
            self.scene_cam_all = json.load(f)
            
        with open(scene_pred_path) as f:
            self.scene_pred_all = json.load(f)

        self.scene_gt_all = self.scene_pred_all 

        # preload dataset
        if cfg.data.preload:
            self.images = self.preload_threading(cfg, self.load_image)
            self.cameras = self.preload_threading(cfg, self.load_camera, data_str="cameras")
            self.obj_mask = self.preload_threading(cfg, self.load_obj_mask)

    def prefetch_all_data(self, cfg):
        assert (not cfg.data.augment)
        # pre-iterate through all samples and group together
        self.all = torch.utils.data._utils.collate.default_collate([s for s in self])

    def load_all_camera_poses(self, cfg, source: str = 'gt') -> torch.Tensor:
        """
        load all camera poses for a given source (gt or predicted).

        Args:
            cfg: The configuration object.
            source: The source of the camera pose (gt or predicted).

        Returns:
            A tensor of shape (num_samples, 4, 4) containing all camera poses.
        """
        # Read the json file at first for this item
        scene_pose_all = self.scene_gt_all if source == 'gt' else self.scene_pred_all
        all_poses = []
        for idx, sample in enumerate(self.list):
            line = sample.split(' ')
            model_name = line[0]
            frame_index = int(line[2])
            obj_scene_id = 0
            rot = scene_pose_all[str(frame_index)][obj_scene_id]['cam_R_m2c']
            trans = scene_pose_all[str(frame_index)][obj_scene_id]['cam_t_m2c']
            pose_raw = torch.eye(4)
            pose_raw[:3, :3] = torch.tensor(rot, dtype=torch.float32).reshape(3, 3)
            pose_raw[:3, 3] = torch.tensor(trans, dtype=torch.float32) / 1000
            all_poses.append(pose_raw)
        all_poses = torch.stack([self.parse_raw_camera(cfg, p) for p in all_poses], dim=0)
        return all_poses

    def __getitem__(self, idx):
        line = self.list[idx].split()
        frame_index = torch.tensor(int(line[2]))

        cfg = self.cfg
        sample = dict(idx=idx)
        obj_scene_id = 0

        image = self.images[idx] if cfg.data.preload else self.load_image(cfg, idx, obj_scene_id=obj_scene_id)
        _, intr, pose_gt, pose_pred = self.cameras[idx] if cfg.data.preload else self.load_camera(cfg, idx,
                                                                                                 obj_scene_id=obj_scene_id)
        z_near, z_far = self.get_range(cfg, idx, obj_scene_id=obj_scene_id)
        obj_mask = self.obj_mask[idx] if cfg.data.preload else self.load_obj_mask(cfg, idx, obj_scene_id=obj_scene_id)
        depth_gt = np.ones_like(obj_mask)

        sample.update(
            image=image,
            intr=intr,
            pose_gt=pose_gt,
            pose_pred=pose_pred,
            z_near=z_near,
            z_far=z_far,
            obj_mask=obj_mask,
            depth_gt=depth_gt,
            frame_index=frame_index,
        )
        if cfg.loss_weight.feat or cfg.gan:
            if self.split == 'train':
                image_syn, mask_syn = self.get_predicted_synthetic_image(cfg, idx, obj_scene_id=obj_scene_id)
                sample.update(image_syn=image_syn, mask_syn=mask_syn)

        if self.split == 'train' and cfg.gan is not None:
            nocs_pred = self.load_predicted_nocs(cfg, idx, obj_scene_id=obj_scene_id)
            normal_pred = self.load_predicted_normal(cfg, idx, obj_scene_id=obj_scene_id)
            sample.update(nocs_pred=nocs_pred, normal_pred=normal_pred)
        return sample

    def load_2d_bbox(self, cfg, idx, obj_scene_id):
        # Parse the split text
        line = self.list[idx].split()
        frame_index = int(line[2])
        assert cfg.H == cfg.W
        bbox = self.scene_info_all[str(frame_index)][obj_scene_id]["bbox_obj"]
        if cfg.data.box_format is None:
            x_ul, y_ul, h, w = bbox
        elif cfg.data.box_format == 'hw':
            x_ul, y_ul, h, w = bbox
        elif cfg.data.box_format == 'wh':
            x_ul, y_ul, w, h = bbox
        else:
            raise NotImplementedError

        center = np.array([int(y_ul + h / 2), int(x_ul + w / 2)])
        scale = int(1.5 * max(h, w))  # detected bb size
        resize = cfg.H / scale
        # print(bbox)
        return center, scale, resize

    def load_image(self, cfg, idx, ext='.png', obj_scene_id=0):
        # Parse the split text
        line = self.list[idx].split()
        folder = line[1]
        frame_index = int(line[2])
        file = "{:06d}{}".format(frame_index, ext)
        image_fname = os.path.join(self.data_path, folder, 'rgb', file)

        center, scale, resize = self.load_2d_bbox(cfg, idx, obj_scene_id=obj_scene_id)
        # print('idx:', idx)
        # print('frame_index:', frame_index)
        # print('center:', center)
        # print('scale:', scale)
        # exit()
        image = cv2.imread(image_fname, -1)[:, :, [2, 1, 0]]
        image = self.Crop_by_Pad(image, center, scale, cfg.H, channel=3).astype(np.uint8)
        image = torchvision_F.to_tensor(image)
        return image

    def load_predicted_synthetic_image(self, cfg, idx, ext='.png', obj_scene_id=0):
        # Parse the split text
        line = self.list[idx].split()
        folder = line[1]
        frame_index = int(line[2])
        file = "{:06d}{}".format(frame_index, ext)

        if cfg.data.pose_source == 'predicted' and self.split == 'train':
            source = 'rgbsyn_pred'
        else:
            source = 'rgbsyn_GT'
        rgba_fname = os.path.join(self.data_path, folder, source, file)
        rgba = cv2.imread(rgba_fname, -1)
        image = torchvision_F.to_tensor(rgba[..., :3][..., [2, 1, 0]])
        alpha = torch.from_numpy((rgba[..., 3] > 0).astype(np.float32))
        return image, alpha

    def load_predicted_nocs(self, cfg, idx, ext='.png', obj_scene_id=0):
        line = self.list[idx].split()
        folder = line[1]
        frame_index = int(line[2])
        file = "{:06d}{}".format(frame_index, ext)

        if cfg.data.pose_source == 'predicted' and self.split == 'train':
            source = 'nocs_pred'
        else:
            source = 'nocs_GT'
        nocs_fname = os.path.join(self.data_path, folder, source, file)
        nocs = cv2.imread(nocs_fname, -1).astype(np.float32)[..., [2, 1, 0]]
        nocs = self.smooth_geo(nocs / 255)
        nocs = torch.from_numpy(nocs).permute(2, 0, 1)
        return nocs

    def load_predicted_normal(self, cfg, idx, ext='.npz', obj_scene_id=0):
        line = self.list[idx].split()
        folder = line[1]
        frame_index = int(line[2])
        file = "{:06d}{}".format(frame_index, ext)

        if cfg.data.pose_source == 'predicted' and self.split == 'train':
            source = 'normal_pred'
        else:
            source = 'normal_GT'
        normal_fname = os.path.join(self.data_path, folder, source, file)
        normal = np.load(normal_fname, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
        normal = self.smooth_geo(normal)
        normal = torch.from_numpy(normal).permute(2, 0, 1)
        return normal

    def load_depth(self, cfg, idx, ext='.png', obj_scene_id=0):
        line = self.list[idx].split()
        folder = line[1]
        frame_index = int(line[2])
        file = "{:06d}{}".format(frame_index, ext)
        depth_scale = self.scene_cam_all[str(frame_index)]["depth_scale"]

        depth_fname = os.path.join(self.data_path, folder, 'depth', file)
        center, scale, resize = self.load_2d_bbox(cfg, idx, obj_scene_id=obj_scene_id)
        depth = cv2.imread(depth_fname, -1) / 1000.
        depth = self.Crop_by_Pad(depth, center, scale, cfg.H, channel=1).astype(np.float32)
        depth = torch.from_numpy(depth).float().squeeze(-1)
        mask = self.load_obj_mask(cfg, idx, obj_scene_id=obj_scene_id)
        depth *= (cfg.nerf.depth.scale * depth_scale)
        depth *= mask.float()
        return depth

    def load_obj_mask(self, cfg, idx, ext='.png', return_visib=True, return_erode=False, obj_scene_id=0):
        line = self.list[idx].split()
        folder = line[1]
        frame_index = int(line[2])
        file = "{:06d}_{:06d}{}".format(frame_index, obj_scene_id, ext)
        
        center, scale, resize = self.load_2d_bbox(cfg, idx, obj_scene_id=obj_scene_id)

        mask_fname_full = os.path.join(self.data_path, folder, 'mask_pred', file)
        
        mask_full = cv2.imread(mask_fname_full, -1)
        mask_full = self.Crop_by_Pad(mask_full, center, scale, cfg.H, 1, cv2.INTER_LINEAR)
        mask_full = mask_full.astype(np.float32)

        # Visible region
        if self.split == 'train':
            visib_source = 'mask_visib_pred'
            mask_fname_visib = os.path.join(self.data_path, folder, visib_source, file)
            mask_visib = cv2.imread(mask_fname_visib, -1)
            if mask_visib.shape[0] != cfg.H:
                mask_visib = self.Crop_by_Pad(mask_visib, center, scale, cfg.H, 1, cv2.INTER_LINEAR)
            if cfg.data.erode_mask:
                # erode the mask during training
                d_kernel = np.ones((3, 3))
                mask_visib = cv2.erode(mask_visib, kernel=d_kernel, iterations=1)
            mask_visib = mask_visib.astype(np.float32)
        else:
            mask_visib = np.ones_like(mask_full)

        if return_visib:
            if self.split == 'train':
                obj_mask = mask_visib > 0
            else:
                obj_mask = mask_full > 0
        else:
            if self.split == 'train':
                obj_mask = mask_visib > 0
            else:
                obj_mask = mask_full > 0
        obj_mask = obj_mask.astype(np.float32)
        if return_erode:
            obj_mask = cv2.erode(obj_mask, kernel=np.ones((3, 3)), iterations=1)
        obj_mask = torch.from_numpy(obj_mask).squeeze(-1)
        return obj_mask

    def get_range(self, cfg, idx, obj_scene_id=0):
        line = self.list[idx].split()
        folder = line[1]
        frame_index = int(line[2])

        depth_min_bg, depth_max_bg = cfg.nerf.depth.range
        depth_min_bg *= cfg.nerf.depth.scale
        depth_max_bg *= cfg.nerf.depth.scale
        depth_min_bg = torch.Tensor([depth_min_bg]).float().expand(cfg.H * cfg.W)
        depth_max_bg = torch.Tensor([depth_max_bg]).float().expand(cfg.H * cfg.W)
        mask = self.load_obj_mask(cfg, idx, return_visib=False, obj_scene_id=obj_scene_id).float()
        if cfg.nerf.depth.range_source == 'box':
            if cfg.data.pose_source == 'predicted' and self.split in ['train', 'val']:
                box_source = cfg.nerf.depth.box_source
            else:
                box_source = 'gt_box'
            # Get meta-information, pose and intrinsics
            file = "{:06d}{}".format(frame_index, '.npz')

            box_fname = os.path.join(self.data_path, folder, box_source, file)
            
            box_range = np.load(box_fname, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
            box_range = box_range.astype(np.float32).transpose((1, 2, 0))
            center, scale, resize = self.load_2d_bbox(cfg, idx, obj_scene_id=obj_scene_id)
            box_range = self.Crop_by_Pad(box_range, center, scale, cfg.H, channel=2).astype(np.float32)
            box_range = torch.from_numpy(box_range)
            if cfg.nerf.depth.box_mask:
                box_range *= mask[..., None]
            box_range = box_range.permute(2, 0, 1).view(2, cfg.H * cfg.W)
            box_range = (box_range / 1000) * cfg.nerf.depth.scale

            z_near, z_far = box_range[0], box_range[1]
            z_near = torch.where(z_near > 0, z_near, depth_min_bg)
            z_far = torch.where(z_far > 0, z_far, depth_max_bg)

        elif cfg.nerf.depth.range_source == 'render':
            depth_gt = self.load_depth(cfg, idx).view(cfg.H * cfg.W)
            z_near, z_far = depth_gt * 0.8, depth_gt * 1.2
            z_near = torch.where(z_near > 0, z_near, depth_min_bg)
            z_far = torch.where(z_far > 0, z_far, depth_max_bg)
            # print(box_fname)
            
        elif cfg.nerf.depth.range_source is None:
            z_near = depth_min_bg
            z_far = depth_max_bg
        else:
            raise NotImplementedError

        assert z_near.shape[0] == cfg.H * cfg.W and z_far.shape[0] == cfg.H * cfg.W
        return z_near, z_far

    def load_camera(self, cfg, idx, obj_scene_id=0):
        # Parse the split text
        line = self.list[idx].split()
        frame_index = int(line[2])

        center, scale, resize = self.load_2d_bbox(cfg, idx, obj_scene_id=obj_scene_id)
        center_offset = self.get_center_offset(center, scale, self.img_H, self.img_W)
        cam_K = self.scene_cam_all[str(frame_index)]["cam_K"]
        cam_K = torch.from_numpy(np.array(cam_K, dtype=np.float32).reshape(3, 3))
        cam_intr = cam_K.clone()
        intr = self.preprocess_intrinsics(cam_intr, resize, center + center_offset, res=cfg.H)

        # Load GT pose if required
        rot = self.scene_gt_all[str(frame_index)][obj_scene_id]['cam_R_m2c']
        trans = self.scene_gt_all[str(frame_index)][obj_scene_id]['cam_t_m2c']
        pose_gt_raw = torch.eye(4)
        pose_gt_raw[:3, :3] = torch.from_numpy(np.array(rot).reshape(3, 3).astype(np.float32))
        pose_gt_raw[:3, 3] = torch.from_numpy(np.array(trans).astype(np.float32)) / 1000  # scale in meter
        pose_gt = self.parse_raw_camera(cfg, pose_gt_raw)

        # Load predicted pose
        if self.split == 'train' and cfg.data.pose_source == 'predicted':
            # TODO: save predicted pose before training
            rot = self.scene_pred_all[str(frame_index)][obj_scene_id]['cam_R_m2c']
            trans = self.scene_pred_all[str(frame_index)][obj_scene_id]['cam_t_m2c']
            pose_pred_raw = torch.eye(4)
            pose_pred_raw[:3, :3] = torch.from_numpy(np.array(rot).reshape(3, 3).astype(np.float32))
            pose_pred_raw[:3, 3] = torch.from_numpy(np.array(trans).astype(np.float32)) / 1000  # scale in meter
            pose_pred = self.parse_raw_camera(cfg, pose_pred_raw) if idx != 0 else pose_gt
        else:
            pose_pred = pose_gt

        return cam_K, intr, pose_gt, pose_pred

    def get_predicted_synthetic_image(self, cfg, idx, ext='.png', obj_scene_id=0):
        # Parse the split text
        line = self.list[idx].split()
        folder = line[1]
        frame_index = int(line[2])
        if not self.multi_obj:
            file = "{:06d}{}".format(frame_index, ext)
        else:
            file = "{:06d}_{:06d}{}".format(frame_index, obj_scene_id, ext)

        if cfg.data.pose_source == 'predicted' and self.split == 'train':
            source = f'rgbsyn_pred'
        else:
            source = 'rgbsyn_GT'
        rgba_fname = os.path.join(self.data_path, folder, source, file)
        rgba = cv2.imread(rgba_fname, -1)
        image = torchvision_F.to_tensor(rgba[..., :3][..., [2, 1, 0]])
        alpha = torch.from_numpy((rgba[..., 3] > 0).astype(np.float32))
        return image, alpha
    
    @staticmethod
    def parse_raw_camera(cfg, pose_raw):
        # Acquire the pose transform point from world to camera
        pose_eye = camera.pose(R=torch.diag(torch.tensor([1, 1, 1])))
        pose = camera.pose.compose([pose_eye, pose_raw[:3]])
        # Scaling
        pose[:, 3] *= cfg.nerf.depth.scale
        return pose

    @staticmethod
    def preprocess_intrinsics(cam_K, resize, crop_center, res):
        # This implementation is tested faithfully. Results in PnP with 0.02% drop.
        K = cam_K

        # First resize from original size to the target size
        K[0, 0] = K[0, 0] * resize
        K[1, 1] = K[1, 1] * resize
        K[0, 2] = (K[0, 2] + 0.5) * resize - 0.5
        K[1, 2] = (K[1, 2] + 0.5) * resize - 0.5

        # Then crop the image --> need to modify the optical center,
        # remember that current top left is the coordinates measured in resized results
        # And its information is vu instead of uv
        top_left = crop_center * resize - res / 2
        K[0, 2] = K[0, 2] - top_left[1]
        K[1, 2] = K[1, 2] - top_left[0]
        return K

    @staticmethod
    def get_center_offset(center, scale, ht, wd):
        upper = max(0, int(center[0] - scale / 2. + 0.5))
        left = max(0, int(center[1] - scale / 2. + 0.5))
        bottom = min(ht, int(center[0] - scale / 2. + 0.5) + int(scale))
        right = min(wd, int(center[1] - scale / 2. + 0.5) + int(scale))

        if upper == 0:
            h_offset = - int(center[0] - scale / 2. + 0.5) / 2
        elif bottom == ht:
            h_offset = - (int(center[0] - scale / 2. + 0.5) + int(scale) - ht) / 2
        else:
            h_offset = 0

        if left == 0:
            w_offset = - int(center[1] - scale / 2. + 0.5) / 2
        elif right == wd:
            w_offset = - (int(center[1] - scale / 2. + 0.5) + int(scale) - wd) / 2
        else:
            w_offset = 0
        center_offset = np.array([h_offset, w_offset])
        return center_offset

    @staticmethod
    # code adopted from CDPN: https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi
    def Crop_by_Pad(img, center, scale, res=None, channel=3, interpolation=cv2.INTER_LINEAR, resize=True):
        # Code from CDPN
        ht, wd = img.shape[0], img.shape[1]

        upper = max(0, int(center[0] - scale / 2. + 0.5))
        left = max(0, int(center[1] - scale / 2. + 0.5))
        bottom = min(ht, int(center[0] - scale / 2. + 0.5) + int(scale))
        right = min(wd, int(center[1] - scale / 2. + 0.5) + int(scale))
        crop_ht = float(bottom - upper)
        crop_wd = float(right - left)

        if resize:
            if crop_ht > crop_wd:
                resize_ht = res
                resize_wd = int(res / crop_ht * crop_wd + 0.5)
            elif crop_ht < crop_wd:
                resize_wd = res
                resize_ht = int(res / crop_wd * crop_ht + 0.5)
            else:
                resize_wd = resize_ht = int(res)

        if channel <= 3:
            tmpImg = img[upper:bottom, left:right]
            if not resize:
                outImg = np.zeros((int(scale), int(scale), channel))
                outImg[int(scale / 2.0 - (bottom - upper) / 2.0 + 0.5):(
                        int(scale / 2.0 - (bottom - upper) / 2.0 + 0.5) + (bottom - upper)),
                int(scale / 2.0 - (right - left) / 2.0 + 0.5):(
                        int(scale / 2.0 - (right - left) / 2.0 + 0.5) + (right - left)), :] = tmpImg
                return outImg

            resizeImg = cv2.resize(tmpImg, (resize_wd, resize_ht), interpolation=interpolation)
            if len(resizeImg.shape) < 3:
                resizeImg = np.expand_dims(resizeImg, axis=-1)  # for depth image, add the third dimension
            outImg = np.zeros((res, res, channel))
            outImg[int(res / 2.0 - resize_ht / 2.0 + 0.5):(int(res / 2.0 - resize_ht / 2.0 + 0.5) + resize_ht),
            int(res / 2.0 - resize_wd / 2.0 + 0.5):(int(res / 2.0 - resize_wd / 2.0 + 0.5) + resize_wd), :] = resizeImg

        else:
            raise NotImplementedError
        return outImg

    @staticmethod
    # code adopted from GDRN: https://github.com/THU-DA-6D-Pose-Group/GDR-Net
    def get_edge(mask, bw=1, out_channel=3):
        if len(mask.shape) > 2:
            channel = mask.shape[2]
        else:
            channel = 1
        if channel == 3:
            mask = mask[:, :, 0] != 0
        edges = np.zeros(mask.shape[:2])
        edges[:-bw, :] = np.logical_and(mask[:-bw, :] == 1, mask[bw:, :] == 0) + edges[:-bw, :]
        edges[bw:, :] = np.logical_and(mask[bw:, :] == 1, mask[:-bw, :] == 0) + edges[bw:, :]
        edges[:, :-bw] = np.logical_and(mask[:, :-bw] == 1, mask[:, bw:] == 0) + edges[:, :-bw]
        edges[:, bw:] = np.logical_and(mask[:, bw:] == 1, mask[:, :-bw] == 0) + edges[:, bw:]
        if out_channel == 3:
            edges = np.dstack((edges, edges, edges))
        return edges

    def smooth_geo(self, x):
        """smooth the edge areas to reduce noise."""
        x = np.asarray(x, np.float32)
        x_blur = cv2.medianBlur(x, 3)
        edges = self.get_edge(x)
        x[edges != 0] = x_blur[edges != 0]
        return x


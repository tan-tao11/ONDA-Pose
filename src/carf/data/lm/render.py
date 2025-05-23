import numpy as np
import cv2
import os
import torch
import json
import torchvision.transforms.functional as torchvision_F
from .. import base
from src.carf.utils import camera

class Dataset(base.Dataset):
    def __init__(self, cfg, split="train", subset=None, multi_obj=False):
        self.raw_H, self.raw_W = 480, 640
        super().__init__(cfg, split)
        assert cfg.H / self.raw_H == cfg.W / self.raw_W

        self.data_path = os.path.join(cfg.data.root, cfg.data.dataset)  # Data path to the bop dataset
        self.split_path = os.path.join("splits", cfg.data.dataset, str(cfg.data.object),
                                       cfg.data.scene, "{}.txt".format(split))
        
        with open(self.split_path, "r") as f:
            self.list = f.readlines()
        self.list = [line.strip() for line in self.list]

        line = self.list[0].split(' ')
        model_name, folder = line[0], line[1]
        scene_obj_path = os.path.join(self.data_path, folder, 'scene_object.json')

        # load scene info
        scene_info_path = os.path.join(self.data_path, folder, 'scene_pred_info.json')
        with open(scene_info_path) as f_info:
            self.scene_info_all = json.load(f_info)
            f_info.close()

        # Load camera pose
        scene_pred_path = os.path.join(self.data_path, folder, 'scene_pred.json')
        scene_cam_path = os.path.join(self.data_path, folder, 'scene_camera.json')

        with open(scene_pred_path, "r") as f:
            self.scene_pred_all = json.load(f)
        
        with open(scene_cam_path, "r") as f:
            self.scene_cam_all = json.load(f)
        
        self.scene_gt_all = self.scene_pred_all 

        # preload dataset
        if cfg.data.preload:
            self.images = self.preload_threading(cfg, self.load_image)
            self.cameras = self.preload_threading(cfg, self.load_camera, data_str="cameras")

    def load_image(self, cfg, idx, ext='.png', obj_scene_id=0):
        # Parse the split text
        line = self.list[idx].split()
        folder = line[1]
        frame_index = int(line[2])
        file = "{:06d}{}".format(frame_index, ext)
        image_fname = os.path.join(self.data_path, folder, 'rgb', file)

        image = cv2.imread(image_fname, -1)[:, :, [2, 1, 0]]
        image = cv2.resize(image, (cfg.W, cfg.H), interpolation=cv2.INTER_LINEAR)
        image = torchvision_F.to_tensor(image)
        return image
    
    def load_camera(self, cfg, idx, obj_scene_id=0):
        # Parse the split text
        line = self.list[idx].split()
        frame_index = int(line[2])

        cam_K = self.scene_cam_all[str(frame_index)]["cam_K"]
        cam_K = torch.from_numpy(np.array(cam_K, dtype=np.float32).reshape(3, 3))
        cam_intr = cam_K.clone()
        resize = cfg.H / self.raw_H 
        intr = self.preprocess_intrinsics(cam_intr, resize)

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
        obj_mask = self.load_obj_mask(cfg, idx, dilate=True, obj_scene_id=obj_scene_id)
        depth_gt = np.ones_like(obj_mask)

        sample.update(
            image=image,
            intr=intr,
            pose_gt=pose_gt,
            pose_pred=pose_pred,
            z_near=z_near,
            z_far=z_far,
            obj_mask=obj_mask,
            frame_index=frame_index,
        )
        return sample

    def load_obj_mask(self, cfg, idx, ext='.png', return_visib=True, return_erode=False, dilate=False, obj_scene_id=0):
        line = self.list[idx].split()
        folder = line[1]
        frame_index = int(line[2])
        file = "{:06d}_{:06d}{}".format(frame_index, obj_scene_id, ext)
        

        mask_fname_full = os.path.join(self.data_path, folder, 'mask_pred', file)
        
        mask_full = cv2.imread(mask_fname_full, -1)
        mask_full = mask_full.astype(np.float32)

        # Visible region
        if self.split == 'train':
            visib_source = 'mask_visib_pred'
            mask_fname_visib = os.path.join(self.data_path, folder, visib_source, file)
            mask_visib = cv2.imread(mask_fname_visib, -1)
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
            obj_mask = cv2.erode(obj_mask, kernel=np.ones((7, 7)), iterations=1)
        if dilate:
            obj_mask = cv2.dilate(obj_mask, kernel=np.ones((7, 7)), iterations=1)
        obj_mask = torch.from_numpy(obj_mask).squeeze(-1)
        return obj_mask
    
    def load_depth(self, cfg, idx, ext='.png', obj_scene_id=0):
        line = self.list[idx].split()
        folder = line[1]
        frame_index = int(line[2])
        file = "{:06d}{}".format(frame_index, ext)
        depth_scale = self.scene_cam_all[str(frame_index)]["depth_scale"]

        depth_fname = os.path.join(self.data_path, folder, 'depth', file)
        depth = cv2.imread(depth_fname, -1) / 1000.
        depth = torch.from_numpy(depth).float().squeeze(-1)
        mask = self.load_obj_mask(cfg, idx, obj_scene_id=obj_scene_id)
        depth *= (cfg.nerf.depth.scale * depth_scale)
        depth *= mask.float()
        return depth
    
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
            if cfg.data.pose_source == 'predicted':
                box_source = cfg.nerf.depth.box_source
            else:
                box_source = 'gt_box'
            # Get meta-information, pose and intrinsics
            file = "{:06d}{}".format(frame_index, '.npz')

            box_fname = os.path.join(self.data_path, folder, box_source, file)
            
            box_range = np.load(box_fname, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
            box_range = box_range.astype(np.float32).transpose((1, 2, 0))
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
    
    @staticmethod
    def parse_raw_camera(cfg, pose_raw):
        # Acquire the pose transform point from world to camera
        pose_eye = camera.pose(R=torch.diag(torch.tensor([1, 1, 1])))
        pose = camera.pose.compose([pose_eye, pose_raw[:3]])
        # Scaling
        pose[:, 3] *= cfg.nerf.depth.scale
        return pose

    @staticmethod
    def preprocess_intrinsics(cam_K, resize):
        # This implementation is tested faithfully. Results in PnP with 0.02% drop.
        K = cam_K

        # First resize from original size to the target size
        K[0, 0] = K[0, 0] * resize
        K[1, 1] = K[1, 1] * resize
        K[0, 2] = (K[0, 2] + 0.5) * resize - 0.5
        K[1, 2] = (K[1, 2] + 0.5) * resize - 0.5
        return K

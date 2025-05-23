import torch
import PIL
import os
import logging
import json
import threading, queue
import torch.nn.functional as F
import torchvision.transforms.functional as torchvision_F
import numpy as np

from tqdm import tqdm

logger = logging.getLogger(__name__)

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, cfg, split="train"):
        super().__init__()
        self.img_H, self.img_W = 480, 640
        self.cfg = cfg
        self.split = split
        self.augment = split == "train" and cfg.data.augment

        self.crop_H, self.crop_W = self.img_H, self.img_W

        if not cfg.W or not cfg.H:
            cfg.W, cfg.H = self.crop_W, self.crop_H 

    def setup_loader(self, cfg, shuffle=False, drop_last=False):
        loader = torch.utils.data.DataLoader(
            self,
            batch_size=cfg.batch_size or 1,
            shuffle=shuffle,
            num_workers=cfg.data.num_workers,
            drop_last=drop_last,
            pin_memory=False,
        )
        logger.info(f"number of samples: {len(self)}")
        return loader
    
    def get_list(self, cfg):
        raise NotImplementedError
    
    def preload_worker(self, data_list, load_func, q, lock, idx_tqdm):
        while True:
            idx = q.get()
            data_list[idx] = load_func(self.cfg, idx)
            with lock:
                idx_tqdm.update()
            q.task_done()

    def preload_threading(self, cfg, load_func, data_str="images"):
        data_list = [None] * len(self)
        q = queue.Queue(maxsize=len(self)) 
        idx_tqdm = tqdm(range(len(self)), desc="preloading {}".format(data_str), leave=False)
        for i in range(len(self)): q.put(i)
        lock = threading.Lock()
        for ti in range(cfg.data.num_workers):
            t = threading.Thread(target=self.preload_worker,
                                 args=(data_list, load_func, q, lock, idx_tqdm), daemon=True)
            t.start()
        q.join()
        idx_tqdm.close()
        assert (all(map(lambda x: x is not None, data_list)))
        return data_list

    def __getitem__(self, idx):
        raise NotImplementedError

    def get_image(self, opt, idx):
        raise NotImplementedError
    
    def preprocess_image(self, cfg, image, is_tensor=False, aug=None):
        if aug is not None:
            raise NotImplementedError
        
        # resize
        if cfg.data.image_size[0] is not None:
            if not is_tensor:
                image = image.resize((cfg.W, cfg.H))
                image = torchvision_F.to_tensor(image)
            else:
                image = F.interpolate(image[None], (cfg.W, cfg.H), mode='nearest')[0]
        return image
    
    def get_all_camera_poses(self, cfg, source='gt'):
        # Read the json file at first for this item
        scene_pose_all = self.scene_gt_all if source == 'gt' else self.scene_pred_all
        pose_raw_all = []
        for idx, sample in enumerate(self.list):
            line = sample.split(' ')
            model_name = line[0]
            frame_index = int(line[2])
            obj_scene_id = 0
            rot = scene_pose_all[str(frame_index)][obj_scene_id]['cam_R_m2c']
            trans = scene_pose_all[str(frame_index)][obj_scene_id]['cam_t_m2c']
            pose_raw = torch.eye(4)
            pose_raw[:3, :3] = torch.from_numpy(np.array(rot).reshape(3, 3).astype(np.float32))
            pose_raw[:3, 3] = torch.from_numpy(np.array(trans).astype(np.float32)) / 1000
            pose_raw_all.append(pose_raw)
        pose_canon_all = torch.stack([self.parse_raw_camera(cfg, p) for p in pose_raw_all], dim=0)
        return pose_canon_all
    
    def preprocess_camera(self, cfg, intr, aug=None):
        # resize
        intr[0] *= cfg.W / self.crop_W
        intr[1] *= cfg.H / self.crop_H
        return intr


    def __len__(self):
        return len(self.list)
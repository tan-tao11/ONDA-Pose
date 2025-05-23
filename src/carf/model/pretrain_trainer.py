import torch
import lpips
import importlib
import logging
import tqdm
import time
from easydict import EasyDict as edict
from . import base
from src.carf.utils import util, util_vis, camera
from ..layers.nerf import NeRF

logger = logging.getLogger(__name__)

class Carf_Step(base.Trainer):
    def __init__(self, cfg):
        
        super().__init__(cfg)
        self.lpips_loss = lpips.LPIPS(net="alex").to(cfg.device)
    
    def load_dataset(self, cfg, eval_split="val", render=False):
        # Get the dataset module
        dataset_name = cfg.data.dataset
        data = importlib.import_module(f"src.carf.data.{dataset_name}.pretrain")

        # Load the training data
        logger.info("Loading training data...")
        self.train_data = data.Dataset(cfg, split="train")
        self.train_loader = self.train_data.setup_loader(cfg, shuffle=True, drop_last=True)

        # Load the test data
        logger.info("Loading test data...")
        if cfg.data.val_on_test:
            eval_split = "test"
        self.test_data = data.Dataset(cfg, split=eval_split)
        self.test_loader = self.test_data.setup_loader(cfg, shuffle=False, drop_last=False)

        # Preload all the data
        self.train_data.prefetch_all_data(cfg)

        # Move the data to the device
        self.train_data.all = edict(util.move_to_device(self.train_data.all, cfg.device))
    
    def train(self, cfg):
        # before training
        logger.info("TRAINING START")
        self.timer = edict(start=time.time(), it_mean=None)
        self.graph.train()
        self.epoch = 0  # dummy for timer
        # training
        if self.iter_start == 0: self.validate(cfg, 0)
        loader = tqdm.trange(cfg.max_iter, desc="training", leave=False)
        for self.iter in loader:
            if self.iter < self.iter_start: continue
            # set var to all available images
            var = self.train_data.all
            self.train_iteration(cfg, var, loader)
            if cfg.optimizer.scheduler: self.scheduler.step()
            if self.iter % cfg.freq.val == 0: self.validate(cfg, self.iter)
            if self.iter % cfg.freq.ckpt == 0: self.save_checkpoint(cfg, epoch=None, iter=self.iter)
        # after training
        if cfg.tb:
            self.tb.flush()
            self.tb.close()
        if cfg.visdom: self.vis.close()
        logger.info("TRAINING DONE")

    @torch.no_grad()
    def log_scalars(self, cfg, var, loss, metric=None, step=0, split="train"):
        super().log_scalars(cfg, var, loss, metric=metric, step=step, split=split)
        # log learning rate
        if split == "train":
            lr = self.optimizer.param_groups[0]["lr"]
            self.tb.add_scalar("{0}/{1}".format(split, "lr"), lr, step)

        # compute PSNR
        psnr = -10 * loss.render.log10()
        self.tb.add_scalar("{0}/{1}".format(split, "PSNR"), psnr, step)
    
    @torch.no_grad()
    def visualize(self, cfg, var, step=0, split="train", eps=1e-10):
        if cfg.tb:
            util_vis.tb_image(cfg, self.tb, step, split, "image", var.image)
            gt_mask = var.obj_mask.view(-1, cfg.H, cfg.W, 1).permute(0, 3, 1, 2).float()  # [B,1,H,W]
            if cfg.data.erode_mask_loss:
                erode_mask = var.erode_mask.view(-1, cfg.H, cfg.W, 1).permute(0, 3, 1, 2).float()  # [B,1,H,W]
                util_vis.tb_image(cfg, self.tb, step, split, "image_masked", var.image * erode_mask + (1 - erode_mask))
            else:
                util_vis.tb_image(cfg, self.tb, step, split, "image_masked", var.image * gt_mask + (1 - gt_mask))
            util_vis.tb_image(cfg, self.tb, step, split, "gt_mask", gt_mask)
            util_vis.tb_image(cfg, self.tb, step, split, "z_near",
                              var.z_near.float().view(-1, cfg.H, cfg.W, 1).permute(0, 3, 1, 2),
                              from_range=(0.9 * cfg.nerf.depth.scale,  var.z_near.max()), cmap='plasma')
            if split == 'train':
                # samples_obj = util_vis.draw_samples(cfg, var, 'object')
                # util_vis.tb_image(cfg, self.tb, step, split, "samples_object", samples_obj, cmap='viridis')
                # samples_bg = util_vis.draw_samples(cfg, var, 'background')
                # util_vis.tb_image(cfg, self.tb, step, split, "samples_background", samples_bg, cmap='viridis')
                util_vis.tb_image(cfg, self.tb, step, split, "depth_gt", var.depth_gt[:, None],
                                  from_range=(0.7 * cfg.nerf.depth.scale, var.depth_gt.max()), cmap='plasma')

            if not cfg.nerf.rand_rays or split != "train":
                mask = (var.obj_mask > 0).view(-1, cfg.H, cfg.W, 1).permute(0, 3, 1, 2).float()

                rgb_map = var.rgb.view(-1, cfg.H, cfg.W, 3).permute(0, 3, 1, 2)  # [B,3,H,W]
                depth_map = var.depth.view(-1, cfg.H, cfg.W, 1).permute(0, 3, 1, 2)  # [B,3,H,W]
                pred_mask = var.opacity.view(-1, cfg.H, cfg.W, 1).permute(0, 3, 1, 2).clamp(0, 1)  # [B,1,H,W]
                gt_mask = var.obj_mask.view(-1, cfg.H, cfg.W, 1).permute(0, 3, 1, 2).float()  # [B,1,H,W]
                depth_error = (depth_map - var.depth_gt[:, None]).abs().view(-1, cfg.H, cfg.W, 1).permute(0, 3, 1, 2)
                depth_error *= mask

                util_vis.tb_image(cfg, self.tb, step, split, "rgb", rgb_map)

                util_vis.tb_image(cfg, self.tb, step, split, "pred_mask", pred_mask)
                util_vis.tb_image(cfg, self.tb, step, split, "gt_mask", gt_mask)
                util_vis.tb_image(cfg, self.tb, step, split, "depth", depth_map * gt_mask,
                                  from_range=(0.7 * cfg.nerf.depth.scale, depth_map.max()), cmap='plasma')
                util_vis.tb_image(cfg, self.tb, step, split, "depth_gt", var.depth_gt[:, None],
                                  from_range=(0.7 * cfg.nerf.depth.scale, depth_map.max()), cmap='plasma')
                util_vis.tb_image(cfg, self.tb, step, split, "depth_error", depth_error,
                                  from_range=(0, torch.quantile(depth_error, 0.99)), cmap='turbo')   

    

# ============================ computation graph for forward/backprop ============================
class Graph(base.Graph):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.nerf = NeRF(cfg)

    @staticmethod
    def ray_batch_sample(ray_identity, ray_idx):
        assert ray_identity.shape[0] == ray_idx.shape[0]
        B, HW, _ = ray_identity.shape
        B, N_sample = ray_idx.shape
        ray_identity = ray_identity.view(B * HW, -1)
        samples_cumsum = ray_idx + HW * torch.arange(B).cuda().unsqueeze(1)  # No need to reshape
        ray_identity_sample = ray_identity[samples_cumsum].view(B, N_sample, -1)
        return ray_identity_sample
    
    @staticmethod
    def get_ray_idx(cfg, var):
        batch_size = len(var.idx)
        rays_on_bg = cfg.nerf.rand_rays // batch_size
        var.ray_idx = torch.randperm(cfg.H * cfg.W, device=cfg.device)[:rays_on_bg].repeat(batch_size, 1)
        # Put all rays together, render all these rays in one-go
        return var
    
    @staticmethod
    def get_pose(cfg, var, mode=None):
        pose_source = dict(gt=var.pose_gt, predicted=var.pose_pred)
        if mode == 'train':
            return pose_source[cfg.data.pose_source]
        else:
            return pose_source['gt']
    
    def forward(self, cfg, var, mode=None):
        pose = self.get_pose(cfg, var, mode=mode)
        depth_min, depth_max = var.z_near, var.z_far
        depth_range = (depth_min[:, :, None], depth_max[:, :, None])  # (B, HW, 1)
        # Do rendering
        if cfg.nerf.rand_rays and mode in ["train", "test-optim"]:
            var = self.get_ray_idx(cfg, var)
            ret = self.render(cfg,
                              pose,
                              intr=var.intr,
                              ray_idx=var.ray_idx,
                              depth_range=depth_range,
                              mode=mode)  # [B,HW,3],[B,HW,1]
            
        else:
            object_mask = var.obj_mask
            ret = self.render_by_slices(cfg,
                                        pose,
                                        intr=var.intr,
                                        depth_range=depth_range,
                                        object_mask=object_mask,
                                        mode=mode)
        var.update(ret)
        return var

    def compute_loss(self, cfg, var, mode=None):
        loss = edict()
        batch_size = len(var.idx)
        HW = cfg.H * cfg.W
        image = var.image.view(batch_size, 3, cfg.H * cfg.W).permute(0, 2, 1)
        # if mode != 'train':
        #     img = var.image.view(batch_size, 3, cfg.H, cfg.W)
        #     rgb = var.rgb.view(batch_size, cfg.H, cfg.W, 3).permute(0, 3, 1, 2)
        #     util.visual_img(rgb[:1], 'real_mask0', self.vis)
        #     util.visual_img(img[:1], 'gt_mask0', self.vis)
        if cfg.nerf.rand_rays and mode in ["train", "test-optim"]:
            image = image.contiguous().view(batch_size, HW, 3)  # B x HW x 3
            image = self.ray_batch_sample(image, var.ray_idx)  # B x HW x 3

        # Compute mask loss
        if cfg.loss_weight.mask is not None:
            gt_mask = var.obj_mask.view(batch_size, cfg.H * cfg.W, 1).float()  # [B, HW, 1]
            # if mode != 'train':
            #     mask = var.obj_mask.view(batch_size, 1, cfg.H, cfg.W).float()
            #     mask_ren = var.opacity[:,:].view(batch_size, 1, cfg.H, cfg.W).float()
            #     util.visual_img(mask[:1], 'mask0', self.vis)
            #     util.visual_img(mask_ren[:1], 'mask_ren0', self.vis)
            if cfg.nerf.rand_rays and mode in ["train", "test-optim"]:
                gt_mask = self.ray_batch_sample(gt_mask, var.ray_idx)
            loss.mask = self.MSE_loss(gt_mask, var.opacity)

        if cfg.loss_weight.reg is not None:
            loss.reg = var.density.mean()

        if cfg.loss_weight.depth is not None:
            if cfg.data.erode_mask_loss:
                mask_obj = var.erode_mask.view(batch_size, cfg.H * cfg.W, 1)
            else:
                mask_obj = var.obj_mask.view(batch_size, cfg.H * cfg.W, 1)

            if mode == 'train':
                depth_gt = self.ray_batch_sample(var.depth_gt.view(batch_size, cfg.H * cfg.W)[..., None],
                                                 var.ray_idx)
                mask_obj = self.ray_batch_sample(mask_obj, var.ray_idx)
            else:
                depth_gt = var.depth_gt.view(batch_size, cfg.H * cfg.W)[..., None]
                mask_obj = mask_obj

            depth_pred = var.depth
            loss.depth = self.scale_invariant_depth_loss(depth_pred, depth_gt, mask_obj)

        if cfg.loss_weight.render is not None:
            if cfg.nerf.mask_obj:
                if cfg.data.erode_mask_loss:
                    mask_obj = var.erode_mask.view(batch_size, cfg.H * cfg.W, 1)
                else:
                    mask_obj = var.obj_mask.view(batch_size, cfg.H * cfg.W, 1)
                if mode == 'train':
                    ray_mask_sample = self.ray_batch_sample(mask_obj, var.ray_idx)  # B x HW x 1
                else:
                    ray_mask_sample = mask_obj
                loss.render = (ray_mask_sample * (image - var.rgb) ** 2).sum() / \
                              (ray_mask_sample.sum() + 1e-5)
                # loss.render = ((image - var.rgb) ** 2).sum() / \
                #               (ray_mask_sample.sum() + 1e-5)
            else:
                loss.render = self.MSE_loss(var.rgb, image)

        return loss   

    def render(self,
               cfg,
               pose,
               intr=None,
               ray_idx=None,
               depth_range=None,
               mode=None):
        batch_size = len(pose)
        center, ray = camera.get_center_and_ray(cfg,
                                                pose,
                                                intr=intr)  # [B,HW,3], center and ray measured in world frame
        while ray.isnan().any():
            center, ray = camera.get_center_and_ray(cfg, pose, intr=intr)  # [B,HW,3]
        # ray center and direction indexing
        center = self.ray_batch_sample(center, ray_idx)
        ray = self.ray_batch_sample(ray, ray_idx)

        # Depth range indexing
        depth_min, depth_max = depth_range[0], depth_range[1]
        depth_min_sample = self.ray_batch_sample(depth_min, ray_idx).squeeze(-1)
        depth_max_sample = self.ray_batch_sample(depth_max, ray_idx).squeeze(-1)
        depth_range = (depth_min_sample, depth_max_sample)

        if cfg.camera.ndc:
            # convert center/ray representations to NDC
            center, ray = camera.convert_NDC(cfg, center, ray, intr=intr)

        # render with main MLP
        depth_samples = self.sample_depth(cfg,
                                          batch_size,
                                          depth_range,
                                          num_rays=ray.shape[1])  # [B,HW,N,1]

        rgb_samples, density_samples = self.nerf.forward_samples(cfg, center, ray, depth_samples, mode=mode)

        rgb, depth, opacity, prob = self.nerf.composite(cfg, ray, rgb_samples, density_samples, depth_samples)
        ret = edict(rgb=rgb, depth=depth, opacity=opacity, density = density_samples)  # [B,HW,K]

        return ret
    
    def render_by_slices(self,
                         cfg,
                         pose,
                         intr=None,
                         depth_range=None,
                         object_mask=None,
                         mode=None):
        ret_all = edict(rgb=[], depth=[], opacity=[], density=[])

        # Acquire the index of the pixels on the objects
        ray_idx_obj = (object_mask.view(cfg.H * cfg.W) > 0).nonzero(as_tuple=True)[0]
        for c in range(0, cfg.H * cfg.W, cfg.nerf.rand_rays):
            ray_idx = torch.arange(c, min(c + cfg.nerf.rand_rays, cfg.H * cfg.W), device=cfg.device)
            ret = self.render(cfg,
                              pose,
                              intr=intr,
                              ray_idx=ray_idx[None],
                              depth_range=depth_range,
                              mode=mode)  # [B,R,3],[B,R,1]
            for k in ret:
                ret_all[k].append(ret[k])
        # group all slices of images
        for k in ret_all:
            ret_all[k] = torch.cat(ret_all[k], dim=1)
        return ret_all
    
    @staticmethod
    def sample_depth(cfg, batch_size, depth_range, num_rays=None):
        depth_min, depth_max = depth_range
        depth_min = depth_min[:, :, None, None]
        depth_max = depth_max[:, :, None, None]
        num_rays = num_rays or cfg.H * cfg.W

        # [B,HW,N,1]
        rand_samples = torch.rand(batch_size, num_rays, cfg.nerf.sample_intvs, 1,
                                  device=cfg.device) if cfg.nerf.sample_stratified else 0.5

        # [B,HW,N,1], Stratified sampling
        rand_samples += torch.arange(cfg.nerf.sample_intvs, device=cfg.device)[None, None, :, None].float()

        # Linearly interpolation
        depth_samples = rand_samples / cfg.nerf.sample_intvs * (depth_max - depth_min) + depth_min  # [B,HW,N,1]

        depth_samples = dict(
            metric=depth_samples,
            inverse=1 / (depth_samples + 1e-8),
        )[cfg.nerf.depth.param]
        # print(depth_samples)
        # exit(0)
        return depth_samples
import numpy as np
import os, sys, time
import torch.nn.functional as torch_F
import torchvision.transforms.functional as torchvision_F
import tqdm
import torch
import lpips
import cv2
import logging
from ..utils import util, util_vis
from ..utils import camera
from . import base
from ..layers.nerf_static_transient_light import NeRF
import importlib
from ..utils.ray_sampler import RaySampler
from ..utils.patch_sampler import FlexPatchSampler
from ..layers.perceptual_loss import PerceptualLoss
from ..layers.discriminator import Discriminator
from torch import autograd
from ..external.pohsun_ssim import pytorch_ssim
from easydict import EasyDict as edict

logger = logging.getLogger(__name__)

class Carf_Step(base.Trainer):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.lpips_loss = lpips.LPIPS(net="alex").to(cfg.device)
    
    def load_dataset(self, cfg, eval_split="val", render=False):
        dataset_name = cfg.data.dataset
        if render:
            data = importlib.import_module(f"src.carf.data.{dataset_name}.render")
        else:
            data = importlib.import_module(f"src.carf.data.{dataset_name}.texture_train")
        logger.info("loading training data...")

        self.train_data = data.Dataset(cfg, split="train")

        self.train_loader = self.train_data.setup_loader(cfg, shuffle=True, drop_last=True)
        logger.info("loading test data...")
        if cfg.data.val_on_test:
            eval_split = "test"
        self.test_data = data.Dataset(cfg, split=eval_split)
        self.test_loader = self.test_data.setup_loader(cfg, shuffle=False)  
        self.train_data.prefetch_all_data(cfg)
        self.train_data.all = edict(util.move_to_device(self.train_data.all, cfg.device))

    def build_networks(self, cfg):
        super().build_networks(cfg)  # include parameters from graph (nerf ...)
        if cfg.nerf.N_anchor_pose is not None:
            self.graph.latent_vars_trans = torch.nn.Embedding(cfg.nerf.N_anchor_pose, cfg.nerf.N_latent_trans).to(cfg.device)
            torch.nn.init.normal_(self.graph.latent_vars_trans.weight)
            self.graph.latent_vars_light = torch.nn.Embedding(cfg.nerf.N_anchor_pose, cfg.nerf.N_latent_light).to(cfg.device)
            torch.nn.init.normal_(self.graph.latent_vars_light.weight)
        else:
            self.graph.latent_vars_trans = torch.nn.Embedding(len(self.train_data), cfg.nerf.N_latent_trans).to(cfg.device)
            torch.nn.init.normal_(self.graph.latent_vars_trans.weight)
            self.graph.latent_vars_light = torch.nn.Embedding(len(self.train_data), cfg.nerf.N_latent_light).to(cfg.device)
            torch.nn.init.normal_(self.graph.latent_vars_light.weight)

    def setup_optimizer(self, cfg):
        logger.info("setting up optimizers...")
        optimizer = getattr(torch.optim, cfg.optimizer.type)

        # set up optimizer for nerf model
        self.optimizer_nerf = optimizer([dict(params=self.graph.nerf.parameters(), lr=cfg.optimizer.lr)])
        self.optimizer_nerf.add_param_group(dict(params=self.graph.latent_vars_light.parameters(), lr=cfg.optimizer.lr))
        self.optimizer_nerf.add_param_group(dict(params=self.graph.latent_vars_trans.parameters(), lr=cfg.optimizer.lr))

        # set up scheduler for nerf model
        if cfg.optimizer.scheduler:
            scheduler = getattr(torch.optim.lr_scheduler, cfg.optimizer.scheduler.type)
            if cfg.optimizer.lr_end:
                assert (cfg.optimizer.scheduler.type == "ExponentialLR")
                if cfg.optimizer.scheduler.gamma is None:
                    cfg.optimizer.scheduler.gamma = (cfg.optimizer.lr_end / cfg.optimizer.lr) ** (1. / cfg.max_epoch)
                else:
                    cfg.optimizer.scheduler.gamma = 0.1 ** (1. / 6000)

            kwargs = {k: v for k, v in cfg.optimizer.scheduler.items() if k != "type"}
            self.scheduler_nerf = scheduler(self.optimizer_nerf, **kwargs)

        # set up optimizer for discriminator
        if cfg.optimizer_disc.type is not None and cfg.gan is not None:
            optimizer_disc = getattr(torch.optim, cfg.optimizer_disc.type)
            self.optimizer_disc = optimizer_disc([dict(params=self.graph.discriminator.parameters(), lr=cfg.optimizer_disc.lr)])

    def restore_pretrained_checkpoint(self, cfg):
        epoch_start, iter_start = None, None
        if cfg.resume_pretrain:
            logger.info("resuming from previous checkpoint...")
            epoch_start, iter_start = util.restore_pretrain_partial_checkpoint(cfg, self, resume=cfg.resume_pretrain)
        elif cfg.resume_real:
            logger.info("resuming from previous checkpoint...")
            epoch_start, iter_start = util.restore_pretrain_nerf(cfg, self, resume=cfg.resume_real)

        else:
            logger.info("initializing weights from scratch...")
        self.epoch_start = epoch_start or 0
        self.iter_start = iter_start or 0

    @staticmethod
    def toggle_grad(model, requires_grad):
        for p in model.parameters():
            p.requires_grad_(requires_grad)
    
    def nerf_trainstep(self, cfg, var):
        # toggle graidents
        self.toggle_grad(self.graph.nerf, True)
        self.toggle_grad(self.graph.latent_vars_trans, True)
        self.toggle_grad(self.graph.latent_vars_light, True)
        if cfg.gan is not None:
            self.toggle_grad(self.graph.discriminator, False)

        # zero out gradients
        self.optimizer_nerf.zero_grad()

        # forward pass of the nerf (generator)
        var = self.graph.nerf_forward(cfg, var, mode='train')
        gloss = self.graph.compute_loss(cfg, var, mode="train", train_step='nerf')
        gloss = self.summarize_loss(cfg, var, gloss)
        gloss.all.backward()

        # update parameters of nerf
        self.optimizer_nerf.step()
        return var, gloss

    def disc_trainstep(self, cfg, var):
        # toggle graidents
        self.toggle_grad(self.graph.nerf, False)
        self.toggle_grad(self.graph.latent_vars_trans, False)
        self.toggle_grad(self.graph.latent_vars_light, False)
        self.toggle_grad(self.graph.discriminator, True)

        # zero out gradients
        self.optimizer_disc.zero_grad()

        # forward pass of the discriminator on real data
        var = self.graph.disc_forward(cfg, var,  mode='train')
        dloss = self.graph.compute_loss(cfg, var, mode="train", train_step='disc')
        dloss = self.summarize_loss(cfg, var, dloss)

        # backward for each component
        # multiply the weighting parameter
        dloss.gan_disc_real *= (10 ** float(cfg.loss_weight.gan_disc_real))
        dloss.gan_disc_fake *= (10 ** float(cfg.loss_weight.gan_disc_fake))

        # backward for real patch, determine if gradients regularization is needed
        if cfg.loss_weight.gan_reg_real is not None:
            dloss.gan_disc_real.backward(retain_graph=True)
            regloss_real = self.graph.compute_grad2(cfg, var.d_real_disc, var.patch_real).mean()
            dloss.gan_reg_real = regloss_real
            regloss_real *= (10 ** float(cfg.loss_weight.gan_reg_real))
            regloss_real.backward()
        else:
            dloss.gan_disc_real.backward()

        # backward for fake patch, determine if gradients regularization is needed
        if cfg.loss_weight.gan_reg_fake is not None:
            dloss.gan_disc_fake.backward(retain_graph=True)
            regloss_fake = self.graph.compute_grad2(cfg, var.d_fake_disc, var.patch_fake).mean()
            dloss.gan_reg_fake = regloss_fake
            regloss_fake *= (10 ** float(cfg.loss_weight.gan_reg_fake))
            regloss_fake.backward()
        else:
            dloss.gan_disc_fake.backward()

        # update params of discriminator
        self.optimizer_disc.step()
        
        return var, dloss
    
    def train_iteration(self, cfg, var, loader):
        # before train iteration
        self.timer.it_start = time.time()
        # train iteration
        var = self.graph.get_ray_idx(cfg, var)
        var, gloss = self.nerf_trainstep(cfg, var)

        if cfg.gan is not None:
            var, dloss = self.disc_trainstep(cfg, var)
            self.graph.discriminator.progress.data.fill_(self.iter / cfg.max_iter)
        else:
            dloss = None
        self.graph.patch_sampler.iterations = self.iter

        if (self.iter + 1) % cfg.freq.log == 0:
            if cfg.gan is not None:
                for train_step, loss in zip(['nerf', 'disc'], [gloss, dloss]):
                    self.log_scalars(cfg, var, loss, step=self.iter + 1, split="train", train_step=train_step)
            else:
                self.log_scalars(cfg, var, gloss, step=self.iter + 1, split="train", train_step='nerf')

        if (self.iter + 1) % cfg.freq.vis == 0: self.visualize(cfg, var, step=self.iter + 1, split="train")
        if (self.iter + 1) % cfg.freq.val == 0: self.validate(cfg, self.iter + 1)
        if (self.iter + 1) % cfg.freq.ckpt == 0: self.save_checkpoint(cfg, epoch=self.epoch, iter=self.iter + 1)
        self.iter += 1

        self.timer.it_end = time.time()
        util.update_timer(cfg, self.timer, self.epoch, len(loader))
        return gloss, dloss
    
    def train_epoch(self, cfg, counter):
        # before train epoch
        self.graph.train()

        loader = self.train_loader
        for batch in loader:
            # train iteration
            var = edict(batch)
            var = util.move_to_device(var, cfg.device)
            gloss, dloss = self.train_iteration(cfg, var, loader)

        if cfg.optimizer.scheduler: self.scheduler_nerf.step()
        if dloss is not None:
            counter.set_postfix(epoch=self.epoch, iter=self.iter, nerf="{:.3f}".format(gloss.all), dis = "{:.3f}".format(dloss.all))
        else:
            counter.set_postfix(epoch=self.epoch, it=self.iter, nerf="{:.3f}".format(gloss.all))

    def train(self, cfg):
        if cfg.max_epoch is not None:
            cfg.max_iter = int(cfg.max_epoch * len(self.train_data) // cfg.batch_size)
        super().train(cfg)

    @torch.no_grad()
    def log_scalars(self, cfg, var, loss, metric=None, step=0, split="train", train_step='nerf'):
        super().log_scalars(cfg, var, loss, metric=metric, step=step, split=split)
        if train_step == 'nerf':
            # log learning rate
            if split == "train":
                lr = self.optimizer_nerf.param_groups[0]["lr"]
                patch_scale_min, patch_scale_max = self.graph.patch_sampler.scales_curr
                # log scale
                self.tb.add_scalar("{0}/{1}".format(split, "lr_nerf"), lr, step)
                self.tb.add_scalar("{0}/{1}".format(split, "patch_scale_min"), patch_scale_min, step)
                self.tb.add_scalar("{0}/{1}".format(split, "patch_scale_max"), patch_scale_max, step)

            # compute PSNR
            # mask = var.obj_mask.view(-1, cfg.H * cfg.W, 1)
            # image = var.image.view(-1, 3, cfg.H * cfg.W).permute(0, 2, 1)
        
        elif train_step == 'disc':
            assert split == 'train'
            lr = self.optimizer_disc.param_groups[0]["lr"]
            self.tb.add_scalar("{0}/{1}".format(split, "lr_disc"), lr, step)

    @torch.no_grad()
    def visualize(self, cfg, var, step=0, split="train", eps=1e-10):
        if cfg.tb:
            mask = (var.obj_mask > 0).view(-1, cfg.H, cfg.W, 1).permute(0, 3, 1, 2).float()
            util_vis.tb_image(cfg, self.tb, step, split, "image_masked", var.image * mask)
            util_vis.tb_image(cfg, self.tb, step, split, "image", var.image)
            util_vis.tb_image(cfg, self.tb, step, split, "z_near",
                              var.z_near.float().view(-1, cfg.H, cfg.W, 1).permute(0, 3, 1, 2),
                              from_range=(0.6 * cfg.nerf.depth.scale,  var.z_near.max()), cmap='plasma')

            if split == 'train':
                rgb_sample = var.rgb.view(-1, cfg.patch_size, cfg.patch_size, 3).permute(0, 3, 1, 2)  # [B,3,H,W]
                rgb_sample_synmasked = rgb_sample * var.mask_syn_sample
                util_vis.tb_image(cfg, self.tb, step, split, "image_sample", var.image_sample)
                util_vis.tb_image(cfg, self.tb, step, split, "rgb_sample", rgb_sample)

                if 'image_syn_sample' in var.keys():
                    util_vis.tb_image(cfg, self.tb, step, split, "syn_image_sample", var.image_syn_sample)
                    util_vis.tb_image(cfg, self.tb, step, split, "rgb_sample_synmasked", rgb_sample_synmasked)
                if 'img_syn_lab' in var.keys() and 'rgb_lab' in var.keys():
                    util_vis.tb_image(cfg, self.tb, step, split, "rgb_lab", var.rgb_lab)
                    util_vis.tb_image(cfg, self.tb, step, split, "img_syn_lab", var.img_syn_lab)
                if 'patch_fake' in var.keys() and 'patch_real' in var.keys():
                    util_vis.tb_image(cfg, self.tb, step, split, "patch_fake_masked", var.patch_fake[:, :3])
                    util_vis.tb_image(cfg, self.tb, step, split, "patch_real_masked", var.patch_real[:, :3])
                if cfg.gan is not None and cfg.gan.geo_conditional:
                    assert 'nocs_sample' in var.keys() and 'normal_sample' in var.keys()
                    _, nocs, normal = var.patch_fake.split(3, dim=1)
                    normal_vis = normal * 0.5 + 0.5
                    util_vis.tb_image(cfg, self.tb, step, split, "normal_predicted", normal_vis)
                    util_vis.tb_image(cfg, self.tb, step, split, "nocs_predicted", nocs)

            if not cfg.nerf.rand_rays or split != "train":
                rgb_map = var.rgb.view(-1, cfg.H, cfg.W, 3).permute(0, 3, 1, 2)  # [B,3,H,W]
                depth_map = var.depth.view(-1, cfg.H, cfg.W, 1).permute(0, 3, 1, 2)  # [B,3,H,W]
                pred_mask = var.opacity_static.view(-1, cfg.H, cfg.W, 1).permute(0, 3, 1, 2).clamp(0, 1)  # [B,1,H,W]
                gt_mask = var.obj_mask.view(-1, cfg.H, cfg.W, 1).permute(0, 3, 1, 2).float()  # [B,1,H,W]
                depth_error = (depth_map - var.depth_gt[:, None]).abs().view(-1, cfg.H, cfg.W, 1).permute(0, 3, 1, 2)
                rgb_static_map = var.rgb_static.view(-1, cfg.H, cfg.W, 3).permute(0, 3, 1, 2)  # [B,3,H,W]
                rgb_transient_map = var.rgb_transient.view(-1, cfg.H, cfg.W, 3).permute(0, 3, 1, 2)  # [B,3,H,W]
                uncert_map = var.uncert.view(-1, cfg.H, cfg.W, 1).permute(0, 3, 1, 2)
                color_error = ((rgb_map - var.image * mask) ** 2).mean(dim=1, keepdim=True).float()
                # if cfg.nerf.mask_obj:
                mask = (var.obj_mask > 0).view(-1, cfg.H, cfg.W, 1).permute(0, 3, 1, 2).float()
                depth_map *= mask
                depth_error *= mask
                util_vis.tb_image(cfg, self.tb, step, split, "rgb", rgb_map)
                util_vis.tb_image(cfg, self.tb, step, split, "rgb_static", rgb_static_map)
                util_vis.tb_image(cfg, self.tb, step, split, "rgb_transient", rgb_transient_map)

                util_vis.tb_image(cfg, self.tb, step, split, "pred_mask", pred_mask)
                util_vis.tb_image(cfg, self.tb, step, split, "gt_mask", gt_mask)
                util_vis.tb_image(cfg, self.tb, step, split, "depth", depth_map,
                                  from_range=(0.8 * cfg.nerf.depth.scale, 1.1 * cfg.nerf.depth.scale), cmap='plasma')
                util_vis.tb_image(cfg, self.tb, step, split, "depth_gt", var.depth_gt[:, None],
                                  from_range=(0.8 * cfg.nerf.depth.scale, 1.1 * cfg.nerf.depth.scale), cmap='plasma')
                util_vis.tb_image(cfg, self.tb, step, split, "depth_error", depth_error,
                                  from_range=(0, torch.quantile(depth_error, 0.99)), cmap='turbo')
                util_vis.tb_image(cfg, self.tb, step, split, "color_error", color_error,
                                  from_range=(0, torch.quantile(color_error, 0.95)), cmap='turbo')
                util_vis.tb_image(cfg, self.tb, step, split, "uncert", uncert_map,
                                  from_range=(uncert_map.min(), torch.quantile(uncert_map, q=0.99)), cmap='viridis')
                
    @torch.no_grad()
    def get_all_training_poses(self, cfg):
        # get ground-truth (canonical) camera poses
        pose_GT = self.train_data.get_all_camera_poses(cfg).to(cfg.device)
        return None, pose_GT
    
    @torch.no_grad()
    def evaluate_full(self, cfg, eps=1e-10):
        self.graph.eval()
        loader = tqdm.tqdm(self.test_loader, desc="evaluating", leave=False)
        res = []
        ckpt_num = 'last' if cfg.resume is True else cfg.resume
        if cfg.render.save_path is not None:
            test_path = cfg.render.save_path
        else:
            test_path = "{0}/test_view_{1}".format(cfg.output, ckpt_num)
        os.makedirs(test_path, exist_ok=True)

        for i, batch in enumerate(loader):
            with torch.no_grad():
                var = edict(batch)
                var = util.move_to_device(var, cfg.device)

                var = self.graph.nerf_forward(cfg, var, mode="eval_noalign")
                # evaluate view synthesis
                image = var.image
                rgb_map = var.rgb_static.view(-1, cfg.H, cfg.W, 3).permute(0, 3, 1, 2)  # * var.obj_mask  # [B,3,H,W]
                depth_map = var.depth.view(-1, cfg.H, cfg.W, 1).permute(0, 3, 1, 2)  # [B,1,H,W]
                mask_map = var.obj_mask.view(-1, cfg.H, cfg.W, 1).permute(0, 3, 1, 2)  # [B,1,H,W]
                pred_mask = var.opacity_static.view(-1, cfg.H, cfg.W, 1).permute(0, 3, 1, 2).clamp(0, 1) 
                vis_mask = (torch.norm(rgb_map.permute(0, 2, 3, 1), p=1, dim=3) > 3.e-1)*1.0
                if cfg.data.image_size != [128, 128]:
                    image = torch_F.interpolate(image, size=[480, 640], mode='bilinear', align_corners=False)
                    rgb_map = torch_F.interpolate(rgb_map, size=[480, 640], mode='bilinear', align_corners=False)
                    depth_map = torch_F.interpolate(depth_map, size=[480, 640], mode='bilinear', align_corners=False)
                    mask_map = torch_F.interpolate(mask_map, size=[480, 640], mode='nearest')
                if cfg.data.scene == 'scene_vis':
                    rgb_map = torchvision_F.center_crop(rgb_map, (256, 256))
                    image = torchvision_F.center_crop(image, (256, 256))
                    depth_map = torchvision_F.center_crop(depth_map, (256, 256))
                    mask_map = torchvision_F.center_crop(mask_map, (256, 256))
                    rgb_map = rgb_map * mask_map + torch.ones_like(rgb_map) * (1 - mask_map)
                fused = torch.cat([rgb_map * mask_map, image * mask_map], dim=-1)
                image_masked = image * mask_map
                depth_map /= cfg.nerf.depth.scale  # scale the to-save depth to metric in meter
                depth_vis = util_vis.preprocess_vis_image(cfg, depth_map, from_range=(0.3, 0.5), cmap="plasma")

                psnr = -10 * self.graph.MSE_loss(rgb_map, image_masked).log10().item()
                ssim = pytorch_ssim.ssim(rgb_map, image_masked).item()
                lpips = self.lpips_loss(rgb_map * 2 - 1, image_masked * 2 - 1).item()
                res.append(edict(psnr=psnr, ssim=ssim, lpips=lpips))
                # dump novel views
                frame_idx = str(var.frame_index.cpu().item()).zfill(6)
                print("{}/{}.png".format(test_path, frame_idx))
                torchvision_F.to_pil_image(rgb_map.cpu()[0]).save("{}/{}.png".format(test_path, frame_idx))
                torchvision_F.to_pil_image(pred_mask.cpu()[0]).save("{}/{}_mask.png".format(test_path, frame_idx))
                # torchvision_F.to_pil_image(vis_mask.cpu()[0]).save("{}/{}_mask_vis.png".format(test_path, frame_idx))
                if cfg.data.scene == 'scene_vis':
                    torchvision_F.to_pil_image(image.cpu()[0]).save("{}/syn_{}.png".format(test_path, frame_idx))
                    torchvision_F.to_pil_image(depth_vis.cpu()[0]).save(
                        "{}/depth_vis_{}.png".format(test_path, frame_idx))

        # show results in terminal
        print("--------------------------")
        print("PSNR:  {:8.2f}".format(np.mean([r.psnr for r in res])))
        print("SSIM:  {:8.2f}".format(np.mean([r.ssim for r in res])))
        print("LPIPS: {:8.2f}".format(np.mean([r.lpips for r in res])))
        print("--------------------------")
        # dump numbers to file
        # quant_fname = "{}/quant.txt".format(cfg.output_path)
        # with open(quant_fname, "w") as file:
        #     for i, r in enumerate(res):
        #         file.write("{} {} {} {}\n".format(i, r.psnr, r.ssim, r.lpips))

    @torch.no_grad()
    def validate(self, cfg, epoch=None):
        self.graph.eval()
        loss_val = edict()
        loader = tqdm.tqdm(self.test_loader, desc="validating", leave=False)
        for iter, batch in enumerate(loader):
            var = edict(batch)
            var = util.move_to_device(var, cfg.device)
            var = self.graph.nerf_forward(cfg, var, mode='val')
            loss = self.graph.compute_loss(cfg, var, mode="val", train_step='nerf')
            loss = self.summarize_loss(cfg, var, loss)
            for key in loss:
                loss_val.setdefault(key, 0.)
                loss_val[key] += loss[key] * len(var.idx)
            loader.set_postfix(loss="{:.3f}".format(loss.all))
            if iter == 0: self.visualize(cfg, var, step=epoch, split="val")
        for key in loss_val: loss_val[key] /= len(self.test_data)
        self.log_scalars(cfg, var, loss_val, step=epoch, split="val")
        logger.info("validation loss: {}".format(loss_val))

    @torch.no_grad()
    def generate_videos_synthesis(self, cfg, eps=1e-10):
        raise NotImplementedError

class Graph(base.Graph):

    def __init__(self, cfg):
        super().__init__(cfg)

        # model to be trained
        self.nerf = NeRF(cfg)
        if cfg.gan is not None:
            self.discriminator = Discriminator(cfg)

        # sampling tool
        self.ray_sampler = RaySampler(cfg)
        self.patch_sampler = FlexPatchSampler(cfg, scale_anneal=0.0002)

        # loss tool
        self.perceptual_loss = PerceptualLoss()

    def get_ray_idx(self, cfg, var):
        coords, scales = self.patch_sampler(nbatch=cfg.batch_size, patch_size=cfg.patch_size, device=cfg.device)
        var.ray_idx = coords
        var.ray_scales = scales
        return var
    
    def get_ray_idx_gan(self, cfg, var):
        coords, scales = self.patch_sampler(nbatch=cfg.batch_size_gan, patch_size=cfg.patch_size, device=cfg.device)
        var.ray_idx = coords
        var.ray_scales = scales
        return var
    
    @staticmethod
    def get_pose(cfg, var, mode=None):
        pose_source = dict(gt=var.pose_gt, predicted=var.pose_pred)
        if mode == 'train':
            return pose_source[cfg.data.pose_source]
        else:
            return pose_source['gt']

    @staticmethod
    def sample_geometry(cfg, var, mode=None):
        batch_size = len(var.idx)
        nocs_pred = var.nocs_pred.contiguous()
        normal_pred = var.normal_pred.contiguous()
        obj_mask = (var.obj_mask > 0).float().contiguous().view(batch_size, 1, cfg.H, cfg.W)
        mask_syn = (var.mask_syn > 0).float().contiguous().view(batch_size, 1, cfg.H, cfg.W)

        if mode == 'train':
            batch_size, h, w, _ = var.ray_idx.shape  # [B, h, w, 2]
            obj_mask = torch_F.grid_sample(obj_mask, var.ray_idx, mode='nearest')  # [B, 1, h, w]
            mask_syn = torch_F.grid_sample(mask_syn, var.ray_idx, mode='nearest')  # [B, 1, h, w]
            nocs_pred = torch_F.grid_sample(nocs_pred, var.ray_idx, mode='bilinear', align_corners=True)
            normal_pred = torch_F.grid_sample(normal_pred, var.ray_idx, mode='bilinear', align_corners=True)

        var.nocs_sample = nocs_pred * mask_syn  # [B, 3, h, w]
        var.normal_sample = normal_pred * mask_syn  # [B, 3, h, w]
        return var
    
    def nerf_forward(self, cfg, var, mode=None):
        """Forward pass for the NeRF model."""
        # Get pose
        pose = self.get_pose(cfg, var, mode=mode)

        # Get depth range
        depth_min, depth_max = var.z_near, var.z_far
        depth_range = (depth_min[:, :, None], depth_max[:, :, None])  # (B, HW, 1)

        # Render
        if cfg.nerf.rand_rays and mode == 'train':
            ret = self.render(cfg,
                              pose,
                              intr=var.intr,
                              ray_idx=var.ray_idx,
                              depth_range=depth_range,
                              sample_idx=var.idx,
                              mode=mode)  # [B,HW,3],[B,HW,1]
        elif mode == 'val':
            object_mask = var.obj_mask
            ret = self.render_by_slices(cfg,
                                        pose,
                                        intr=var.intr,
                                        depth_range=depth_range,
                                        object_mask=object_mask,
                                        sample_idx=None,
                                        mode=mode)
        else:
            object_mask = var.obj_mask

            ret = self.render_by_slices(cfg,
                                        pose,
                                        intr=var.intr,
                                        depth_range=depth_range,
                                        object_mask=object_mask,
                                        sample_idx=var.idx[0],
                                        mode=mode)

        # Update var
        var.update(ret)

        # Prepare for GAN
        if mode == 'train' and cfg.gan is not None:
            if cfg.gan.geo_conditional:
                var = self.sample_geometry(cfg, var, mode)
            batch_size, h, w, _ = var.ray_idx.shape  # [B, h, w, 2]
            patch_fake = var.rgb.view(batch_size, h, w, 3).permute(0, 3, 1, 2)  # [B, 3, h, w]

            if cfg.gan.geo_conditional:
                patch_fake = torch.cat([patch_fake, var.nocs_sample, var.normal_sample], dim=1)
            var.d_fake_nerf = self.discriminator(cfg, patch_fake, var.ray_scales)
        return var

    def disc_forward(self, cfg, var, mode):
        """Forward pass for the discriminator."""
        if mode != 'train':
            raise Exception('No use of discriminator in val/testing phase of NeRF!')
        
        # Get the real and fake images
        batch_size, h, w, _ = var.ray_idx.shape
        image_real = var.image_sample
        
        mask = var.mask_sample
        rgb = var.rgb.view(batch_size, h, w, 3).permute(0, 3, 1, 2).contiguous()
        rgb_static = var.rgb_static.view(batch_size, h, w, 3).permute(0, 3, 1, 2).contiguous()
        mask_syn = var.mask_syn_sample
        image_syn = var.image_syn_sample
        
        # Compute the real and fake patches
        mask_pad = torch.logical_and(mask_syn == 0, mask == 1).float()
        patch_real = (image_syn * mask_syn + rgb.detach() * mask_pad).detach()
        # patch_real = image_syn 
        
        patch_fake = rgb * mask

        var.patch_real = patch_real
        var.patch_fake = patch_fake.detach()
        
        # Compute the discriminator output for the real and fake patches
        if cfg.gan.geo_conditional:
            var = self.sample_geometry(cfg, var, mode)
            var.patch_real = torch.cat([var.patch_real, var.nocs_sample, var.normal_sample], dim=1)
            var.patch_fake = torch.cat([var.patch_fake, var.nocs_sample, var.normal_sample], dim=1)
        var.patch_fake.requires_grad_()
        var.patch_real.requires_grad_()
        var.d_real_disc = self.discriminator(cfg, var.patch_real, var.ray_scales)
        var.d_fake_disc = self.discriminator(cfg, var.patch_fake, var.ray_scales)
        
        return var

    def render(self, cfg, pose, intr=None, ray_idx=None, depth_range=None, sample_idx=None, mode=None):
        # Depth range indexing
        # [B,HW,3], center and ray measured in world frame
        if mode == 'train':
            batch_size, h, w, _ = ray_idx.shape  # [B, h, w, 2]
            
            depth_min, depth_max = depth_range[0], depth_range[1]
            center, ray = self.ray_sampler.get_rays(cfg, intrinsics=intr, coords=ray_idx, pose=pose)
            while ray.isnan().any():
                center, ray = self.ray_sampler.get_rays(cfg, intrinsics=intr, coords=ray_idx, pose=pose)
            depth_min_sample, depth_max_sample = self.ray_sampler.get_bounds(cfg, coords=ray_idx,
                                                                             z_near=depth_min, z_far=depth_max)

            # Reshape the output results
            center = center.view(batch_size, h * w, -1)  # [B, hw, 3]
            ray = ray.view(batch_size, h * w, -1)  # [B, hw, 3]
            depth_min_sample = depth_min_sample.view(batch_size, h * w)  # [B, hw]
            depth_max_sample = depth_max_sample.view(batch_size, h * w)  # [B, hw]

        else:
            batch_size = len(pose)
            # [B,HW,3], center and ray measured in world frame
            center, ray = camera.get_center_and_ray(cfg, pose, intr=intr)
            while ray.isnan().any():
                center, ray = camera.get_center_and_ray(cfg, pose, intr=intr)  # [B,HW,3]

            # ray center and direction indexing
            center = self.ray_batch_sample(center, ray_idx)
            ray = self.ray_batch_sample(ray, ray_idx)

            # Depth range indexing
            depth_min, depth_max = depth_range[0], depth_range[1]
            depth_min_sample = self.ray_batch_sample(depth_min, ray_idx).squeeze(-1)
            depth_max_sample = self.ray_batch_sample(depth_max, ray_idx).squeeze(-1)

        if cfg.camera.ndc:
            # convert center/ray representations to NDC
            center, ray = camera.convert_NDC(cfg, center, ray, intr=intr)

        # Acquire depth range
        depth_range = (depth_min_sample, depth_max_sample)
        depth_samples = self.sample_depth(cfg, batch_size, depth_range, num_rays=ray.shape[1])  # [B, HW, N, 1]

        # Prepare latent variables
        if mode == 'train':
            latent_variable_trans = self.latent_vars_trans.weight[sample_idx]
            latent_variable_light = self.latent_vars_light.weight[sample_idx]
        elif mode == 'val':
            latent_variable_trans = self.latent_vars_trans.weight[0][None]
            latent_variable_light = self.latent_vars_light.weight[0][None]
        else:
            # Not sure about the situation for the other object, but for cat,
            # this zero latent variable is important!!
            if cfg.render.transient == 'zero':
                latent_variable_trans = torch.zeros(batch_size, cfg.nerf.N_latent_trans).cuda()
            elif cfg.render.transient == 'sample':
                latent_variable_trans = self.latent_vars_trans.weight[sample_idx][None]
            else:
                raise NotImplementedError
            # latent_variable_light = self.latent_vars_light.weight[sample_idx][None]
            latent_variable_light = self.latent_vars_light.weight[0][None]
        # Forward samples to get rgb values and density values
        rgb_samples, density_samples, uncert_samples = self.nerf.forward_samples(cfg,
                                                                                 center=center,
                                                                                 ray=ray,
                                                                                 depth_samples=depth_samples,
                                                                                 latent_variable_trans=latent_variable_trans,
                                                                                 latent_variable_light=latent_variable_light,
                                                                                 mode=mode)

        # composition
        (rgb, rgb_static, rgb_transient, depth, opacity, opacity_static, opacity_transient,
         prob, uncert, alpha_static, alpha_transient) = self.nerf.composite(
            cfg,
            ray,
            rgb_samples,
            density_samples,
            depth_samples,
            uncert_samples)

        # track the results
        ret = edict(rgb=rgb, rgb_static=rgb_static, rgb_transient=rgb_transient,
                    opacity=opacity, opacity_static=opacity_static, opacity_transient=opacity_transient,
                    uncert=uncert, depth=depth, alpha_static=alpha_static, alpha_transient=alpha_transient,
                    density=density_samples)  # [B,HW,K]

        return ret
    
    def render_by_slices(self, cfg, pose, intr=None, depth_range=None, object_mask=None, sample_idx=None, mode=None):
        # Initialize output dictionary
        output = edict(rgb=torch.zeros(1, cfg.H * cfg.W, 3).cuda(),
                       rgb_static=torch.zeros(1, cfg.H * cfg.W, 3).cuda(),
                       rgb_transient=torch.zeros(1, cfg.H * cfg.W, 3).cuda(),
                       opacity=torch.zeros(1, cfg.H * cfg.W, 1).cuda(),
                       opacity_static=torch.zeros(1, cfg.H * cfg.W, 1).cuda(),
                       opacity_transient=torch.zeros(1, cfg.H * cfg.W, 1).cuda(),
                       depth=torch.zeros(1, cfg.H * cfg.W, 1).cuda(),
                       uncert=torch.ones(1, cfg.H * cfg.W, 1).cuda() * cfg.nerf.min_uncert,
                       alpha_static=torch.ones(1, cfg.H * cfg.W, cfg.nerf.sample_intvs).cuda(),
                       alpha_transient=torch.ones(1, cfg.H * cfg.W, cfg.nerf.sample_intvs).cuda(),
                       density=torch.ones(1, cfg.H * cfg.W, cfg.nerf.sample_intvs, 2).cuda())
        
        if mode == 'val':
            for c in range(0, cfg.H * cfg.W, cfg.nerf.rand_rays):
                ray_idx = torch.arange(c, min(c + cfg.nerf.rand_rays, cfg.H * cfg.W), device=cfg.device)[None]
                ret = self.render(cfg,
                                pose,
                                intr=intr,
                                ray_idx=ray_idx,
                                depth_range=depth_range,
                                sample_idx=sample_idx,
                                mode=mode)  # [B,R,3],[B,R,1]
            for k in ret:
                output[k][:, ray_idx] = ret[k][0]
        else:
            # Get the object mask
            ray_idx_obj = (object_mask.view(cfg.H * cfg.W) > 0).nonzero(as_tuple=True)[0]

            # Render slices of the image
            for c in range(0, len(ray_idx_obj), cfg.nerf.rand_rays):
                ray_idx = ray_idx_obj[c:min(c + cfg.nerf.rand_rays, len(ray_idx_obj))][None]
                ret = self.render(cfg,
                                pose,
                                intr=intr,
                                ray_idx=ray_idx,
                                depth_range=depth_range,
                                sample_idx=sample_idx,
                                mode=mode)  # [B,R,3],[B,R,1]
                for k in ret:
                    output[k][:, ray_idx] = ret[k][0]

        return output

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

        depth_samples = dict(metric=depth_samples, inverse=1 / (depth_samples + 1e-8), )[cfg.nerf.depth.param]
        return depth_samples

    @staticmethod
    def ray_batch_sample(ray_identity: torch.Tensor, ray_idx: torch.Tensor) -> torch.Tensor:
        assert ray_identity.shape[0] == ray_idx.shape[0], "Batch size mismatch"
        batch_size, height_width, _ = ray_identity.shape
        batch_size, num_samples = ray_idx.shape
        ray_identity = ray_identity.view(batch_size * height_width, -1)
        samples_cumsum = ray_idx + height_width * torch.arange(batch_size).cuda().unsqueeze(1)
        ray_identity_sample = ray_identity[samples_cumsum].view(batch_size, num_samples, -1)
        return ray_identity_sample
    
    def compute_loss(self, cfg, var, mode=None, train_step='nerf'):
        loss = edict()
        batch_size = len(var.idx)
        rgb = var.rgb
        uncert = var.uncert
        image = var.image.contiguous()
        obj_mask = (var.obj_mask > 0).float().contiguous().view(batch_size, 1, cfg.H, cfg.W)
        if 'image_syn' in var.keys() and 'mask_syn' in var.keys():
            image_syn = var.image_syn.contiguous()
            mask_syn = (var.mask_syn > 0).float().contiguous()
        else:
            image_syn = image
            mask_syn = obj_mask
        rgb_map = None
        if cfg.nerf.rand_rays and mode in ["train", "test-optim"]:
            batch_size, h, w, _ = var.ray_idx.shape  # [B, h, w, 2]
            image = torch_F.grid_sample(image, var.ray_idx, mode='bilinear', align_corners=True)  # [B, 3, h, w]
            obj_mask = torch_F.grid_sample(obj_mask, var.ray_idx, mode='nearest')  # [B, 1, h, w]
            mask_syn = torch_F.grid_sample(mask_syn[:, None, :, :], var.ray_idx, mode='nearest')  # [B, 1, h, w]
            image_syn = torch_F.grid_sample(image_syn, var.ray_idx, mode='bilinear', align_corners=True)
            rgb = rgb.view(batch_size, h, w, 3).permute(0, 3, 1, 2)  # [B, 3, h, w]
            uncert = uncert.view(batch_size, h, w, 1).permute(0, 3, 1, 2)  # [B, 1, h, w]
            rgb_map = var.rgb_static.view(-1, h, w, 3).permute(0, 3, 1, 2)  # * var.obj_mask  # [B,3,H,W]
        else:
            image = image.view(batch_size, 3, cfg.H, cfg.W)
            obj_mask = obj_mask.view(batch_size, 1, cfg.H, cfg.W)
            mask_syn = mask_syn.view(batch_size, 1, cfg.H, cfg.W)
            rgb = rgb.view(batch_size, cfg.H, cfg.W, 3).permute(0, 3, 1, 2)  # [B, 3, H, W]
            uncert = uncert.view(batch_size, cfg.H, cfg.W, 1).permute(0, 3, 1, 2)  # [B, 1, H, W]
            image_syn = image_syn.view(batch_size, 3, cfg.H, cfg.W)
            rgb_map = var.rgb_static.view(-1, cfg.H, cfg.W, 3).permute(0, 3, 1, 2)  # * var.obj_mask  # [B,3,H,W]
        
        var.image_syn_sample = image_syn
        var.image_sample = image
        var.mask_sample = obj_mask
        var.mask_syn_sample = mask_syn
        if train_step == 'nerf':
            if cfg.loss_weight.reg is not None:
                vis_mask = torch.norm(rgb_map.permute(0, 2, 3, 1), p=1, dim=3)
                bias = torch.ones_like(vis_mask)*0.2
                vis_mask = torch.relu(-1*vis_mask+bias)
                loss.reg = (vis_mask*obj_mask[:,0,:,:]).sum()/batch_size
            if cfg.loss_weight.render is not None:
                if cfg.nerf.mask_obj:
                    loss.render = (mask_syn * ((image_syn - rgb) ** 2 / uncert ** 2)).sum() / (mask_syn.sum() + 1e-5)
                else:
                    loss.render = self.MSE_loss(rgb, image)

            # Compute mask loss
            if cfg.loss_weight.mask is not None:
                loss.mask = self.MSE_loss(obj_mask, var.opacity[..., None])

            if cfg.loss_weight.uncert is not None:
                loss.uncert = 5 + torch.log(var.uncert ** 2).mean() / 2

            if cfg.loss_weight.trans_reg is not None:
                loss.trans_reg = var.density[..., -1].mean()

            if cfg.gan is not None and cfg.loss_weight.gan_nerf is not None and mode == 'train':
                loss.gan_nerf = self.compute_gan_loss(cfg, d_outs=var.d_fake_nerf, target=1)

        elif train_step == 'disc':
            # regularization for real patch
            dloss_real = self.compute_gan_loss(cfg, d_outs=var.d_real_disc, target=1)
            dloss_fake = self.compute_gan_loss(cfg, d_outs=var.d_fake_disc, target=0)
            # compute_loss
            if cfg.loss_weight.gan_disc_real is not None:
                loss.gan_disc_real = dloss_real

            if cfg.loss_weight.gan_disc_fake is not None:
                loss.gan_disc_fake = dloss_fake

        else:
            raise NotImplementedError
        return loss

    @staticmethod
    def compute_grad2(cfg, d_outs, x_in):
        d_outs = [d_outs] if not isinstance(d_outs, list) else d_outs
        reg = 0
        for d_out in d_outs:
            batch_size = x_in.size(0)
            grad_dout = autograd.grad(
                outputs=d_out.sum(), inputs=x_in,
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            grad_dout2 = grad_dout.pow(2)
            assert (grad_dout2.size() == x_in.size())
            reg += grad_dout2.view(batch_size, -1).sum(1)
        return reg / len(d_outs)

    @staticmethod
    def compute_gan_loss(cfg, d_outs, target):
        d_outs = [d_outs] if not isinstance(d_outs, list) else d_outs
        loss = torch.tensor(0.0, device=d_outs[0].device)
        for d_out in d_outs:

            targets = d_out.new_full(size=d_out.size(), fill_value=target)

            if cfg.gan.type == 'standard':
                loss += torch_F.binary_cross_entropy_with_logits(d_out, targets)
            elif cfg.gan.type == 'wgan':
                loss += (2 * target - 1) * d_out.mean()
            else:
                raise NotImplementedError

        return loss / len(d_outs)

    def wgan_gp_reg(self, cfg, x_real, x_fake, y, center=1.):
        batch_size = y.size(0)
        eps = torch.rand(batch_size, device=y.device).view(batch_size, 1, 1, 1)
        x_interp = (1 - eps) * x_real + eps * x_fake
        x_interp = x_interp.detach().requires_grad_()
        d_out = self.discriminator(x_interp, y)
        reg = (self.compute_grad2(cfg, d_out, x_interp).sqrt() - center).pow(2).mean()
        return reg

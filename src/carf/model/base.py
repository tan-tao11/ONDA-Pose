import importlib
import torch
import os
import logging
import visdom
import time
import tqdm
import tensorboardX
from easydict import EasyDict as edict
from src.carf.utils import util

logger = logging.getLogger(__name__)

class Trainer():

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        os.makedirs(cfg.output, exist_ok=True)
    
    def setup(self, cfg, eval_split='val', render=False):
        self.load_dataset(cfg, eval_split=eval_split, render=render)
        self.build_networks(cfg)
        self.setup_optimizer(cfg)
        if 'tex' in cfg.model:
            self.restore_pretrained_checkpoint(cfg)
        self.restore_checkpoint(cfg)
        self.setup_visualizer(cfg)

    def load_dataset(self, cfg, eval_split="val"):
        dataset_name = cfg.data.dataset
        data = importlib.import_module(f"src.carf.data.{dataset_name}")
        logger.info(f"Loading training data...")
        self.train_data = data.Dataset(cfg, split="train")
        self.train_loader = self.train_data.setup_loader(cfg, shuffle=True)
        logger.info(f"Loading test data...")
        if cfg.data.val_on_test:
            eval_spliter = "test"
        self.test_data = data.Dataset(cfg, split=eval_split)
        self.test_loader = self.test_data.setup_loader(cfg, shuffle=False)  
    
    def build_networks(self, cfg):
        graph = importlib.import_module(f"src.carf.model.{cfg.model}")
        logger.info(f"Building {cfg.model} model...")
        self.graph = graph.Graph(cfg).to(cfg.device)
    
    def setup_optimizer(self, cfg):
        logger.info(f"Setting up optimizer...")
        optimizer = getattr(torch.optim, cfg.optimizer.type)
        self.optimizer = optimizer([dict(params=self.graph.parameters(), lr=cfg.optimizer.lr)])
        # Setup scheduler
        if cfg.optimizer.scheduler is not None:
            scheduler = getattr(torch.optim.lr_scheduler, cfg.optimizer.scheduler.type)
            kwargs = {k: v for k, v in cfg.optimizer.scheduler.items() if k != "type"}
            self.scheduler = scheduler(self.optimizer, **kwargs)
    
    def restore_checkpoint(self, cfg):
        epoch_start, iter_start = None, None
        if cfg.resume:
            logger.info(f"Restoring checkpoint...")
            epoch_start, iter_start = util.restore_checkpoint(cfg, self, resume=cfg.resume)
        elif cfg.load is not None:
            logger.info(f"Loading checkpoint...")
            epoch_start, iter_start = util.restore_checkpoint(cfg, self, load_name=cfg.load)
        else:
            logger.info("Initializing from scratch...")
        
        self.epoch_start = epoch_start or 0
        self.iter_start = iter_start or 0
    
    def setup_visualizer(self, cfg):
        logger.info(f"Setting up visualizer...")
        if cfg.tb:
            self.tb = tensorboardX.SummaryWriter(log_dir=cfg.output, flush_secs=10)
        if cfg.visdom:
            is_open = util.check_socket_open(cfg.visdom.server, cfg.visdom.port)
            retry = None
            while not is_open:
                retry = input("visdom port ({}) not open, retry? (y/n) ".format(cfg.visdom.port))
                if retry not in ["y", "n"]: continue
                if retry == "y":
                    is_open = util.check_socket_open(cfg.visdom.server, cfg.visdom.port)
                else:
                    break
        # self.vis = visdom.Visdom(server=cfg.visdom.server, port=str(cfg.visdom.port), env=cfg.group)

    def train(self, cfg):
        logger.info(f"Training...")
        self.timer = edict(start=time.time(), it_mean=None)
        self.graph.train()
        self.epoch = 0
        self.iter = self.iter_start

        if self.iter_start == 0: self.validate(cfg, epoch=0)
        counter = tqdm.trange(cfg.max_epoch, desc="training", leave=False)
        for self.epoch in counter:
            if self.epoch < self.epoch_start: continue
            if self.iter < self.iter_start: continue
            self.train_epoch(cfg, counter)
        # after training
        if cfg.tb:
            self.tb.flush()
            self.tb.close()
        if cfg.visdom: self.vis.close()
        logger.info("Training finished.")

    def train_epoch(self, cfg, counter):
        self.graph.train()
        
        loader = self.train_loader
        for batch in loader:
            var = edict(batch)
            var = util.move_to_device(var, cfg.device)
            loss = self.train_iteration(cfg, var, loader)
        
        if cfg.optimizer.scheduler is not None:
            self.scheduler.step()
        if (self.epoch+ 1) % cfg.freq.val == 0:
            self.validate(cfg, epoch=self.epoch + 1)
        if (self.epoch+ 1) % cfg.freq.ckpt == 0:
            self.save_checkpoint(cfg, epoch=self.epoch + 1, iter=self.iter)
        
        counter.set_postfix(epoch=self.epoch)
    
    def train_iteration(self, cfg, var, loader):
        self.timer.iter_start = time.time()

        self.optimizer.zero_grad()
        var = self.graph.forward(cfg, var, mode="train")
        loss = self.graph.compute_loss(cfg, var, mode="train")
        loss = self.summarize_loss(cfg, var, loss)
        loss.all.backward()
        self.optimizer.step()

        if (self.iter + 1) % cfg.freq.log == 0:
            self.log_scalars(cfg, var, loss, step=self.iter + 1, split="train")
        if (self.iter + 1) % cfg.freq.vis == 0:
            self.visualize(cfg, var, step=self.iter + 1, split="train")
        self.iter += 1
        loader.set_postfix(it=self.iter, loss="{:.3f}".format(loss.all))
        self.timer.it_end = time.time()
        util.update_timer(cfg, self.timer, self.epoch, len(loader))
        return loss

    def summarize_loss(self, cfg, var, loss):
        loss_all = 0. 

        for key in loss:
            assert (key in cfg.loss_weight)
            assert (loss[key].shape == ())
            if cfg.loss_weight[key] is not None:
                assert not torch.isinf(loss[key]), "loss {} is Inf".format(key)
                assert not torch.isnan(loss[key]), "loss {} is NaN".format(key)
                loss_all += 10 ** float(cfg.loss_weight[key]) * loss[key]
        loss.update(all=loss_all)
        return loss
    
    @torch.no_grad()
    def validate(self, cfg, epoch=None):
        self.graph.eval()
        loss_val = edict()
        loader = tqdm.tqdm(self.test_loader, desc="validating", leave=False)
        for iter, batch in enumerate(loader):
            var = edict(batch)
            var = util.move_to_device(var, cfg.device)
            var = self.graph.forward(cfg, var, mode="val")
            loss = self.graph.compute_loss(cfg, var, mode="val")
            loss = self.summarize_loss(cfg, var, loss)
            for key in loss:
                loss_val.setdefault(key, 0.)
                loss_val[key] += loss[key] * len(var.idx)
            loader.set_postfix(loss="{:.3f}".format(loss.all))
            if iter == 0: self.visualize(cfg, var, step=epoch, split="val")
        for key in loss_val: loss_val[key] /= len(self.test_data)
        self.log_scalars(cfg, var, loss_val, step=epoch, split="val")
        
        logger.info(f"Validation loss: {loss_val.all:.3f}")

    @torch.no_grad()
    def log_scalars(self, cfg, var, loss, metric=None, step=0, split="train"):
        for key, value in loss.items():
            if key == "all": continue
            if cfg.loss_weight[key] is not None:
                self.tb.add_scalar("{0}/loss_{1}".format(split, key), value, step)
        if metric is not None:
            for key, value in metric.items():
                self.tb.add_scalar("{0}/{1}".format(split, key), value, step)
    
    @torch.no_grad()
    def visualize(self, cfg, var, step=0, split="train"):
        raise NotImplementedError

    def save_checkpoint(self, cfg, epoch=0, iter=0, latest=False):
        util.save_checkpoint(cfg, self, epoch=epoch, iter=iter, latest=latest)
        if not latest:
            logger.info("checkpoint saved: ({}), epoch {} (iteration {})".format(cfg.output, epoch, iter))

# ============================ computation graph for forward/backprop ============================

class Graph(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()

    def forward(self, cfg, var, mode=None):
        raise NotImplementedError
        return var

    def compute_loss(self, cfg, var, mode=None):
        loss = edict()
        raise NotImplementedError
        return loss

    def L1_loss(self, pred, label=0):
        loss = (pred.contiguous() - label).abs()
        return loss.mean()

    def MSE_loss(self, pred, label=0):
        loss = (pred.contiguous() - label) ** 2
        return loss.mean()

    def scale_invariant_depth_loss(self, depth_pred, depth_target, mask=None):
        depth_pair = torch.stack((depth_pred, depth_target), dim=-1)  # [B, HW, C, 2]
        min_depth, _ = torch.min(depth_pair, dim=-1)  # [B, HW, C]
        max_depth, _ = torch.max(depth_pair, dim=-1)  # [B, HW, C]
        loss = 1 - min_depth / (max_depth + 1e-5)
        if mask is not None:
            mask = mask.float()
            loss = (loss * mask).sum() / (mask.sum() + 1e-5)
        return loss

    def point_loss(self, point_pred, point_target, mask):
        e = torch.norm(point_pred - point_target, p=2, dim=-1, keepdim=True)  # B x HW x 1
        c = 2 * torch.quantile(e, q=0.5, dim=1, keepdim=True).detach()  # B x 1 x 1
        loss = -torch.expm1(-0.5 * (e / c) ** 2)
        mask = mask.float()
        loss = (loss * mask).sum() / (mask.sum() + 1e-5)
        return loss

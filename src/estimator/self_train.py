import argparse
import torch
import os
import mmcv
import logging
import time
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import os.path as osp
from .train_utils import batch_data_self
from .utils import solver_utils 
from .utils import common as comm
from .utils.load_model import load_model
from .utils.build_optimizers import build_optimizer
from .data.dataset_factory import register_datasets_in_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from .data.data_loader import build_gdrn_self_train_loader, build_gdrn_test_loader
from .utils.utils import dprint
from .utils.my_checkpoint import MyCheckpointer
from .utils.torch_utils.torch_utils import ModelEMA
from .utils.torch_utils.misc import nan_to_num
from .utils.my_writer import MyCommonMetricPrinter, MyJSONWriter, MyTensorboardXWriter
from .utils.gdrn_custom_evaluator import GDRN_EvaluatorCustom
from .utils.gdrn_evaluator import gdrn_inference_on_dataset
from .train_utils import get_dibr_models_renderer
from . import ref
from train_utils import compute_self_loss
from datetime import timedelta
from collections import OrderedDict
from omegaconf import OmegaConf
from torch.cuda.amp import autocast, GradScaler
from detectron2.checkpoint import PeriodicCheckpointer
from detectron2.utils.events import EventStorage
from src.estimator.utils import logger

def get_tbx_event_writer(out_dir, backup=False):
    tb_logdir = osp.join(out_dir, "tb")
    mmcv.mkdir_or_exist(tb_logdir)
    if backup and comm.is_main_process():
        old_tb_logdir = osp.join(out_dir, "tb_old")
        mmcv.mkdir_or_exist(old_tb_logdir)
        os.system("mv -v {} {}".format(osp.join(tb_logdir, "events.*"), old_tb_logdir))

    tbx_event_writer = MyTensorboardXWriter(tb_logdir, backend="tensorboardX")
    return tbx_event_writer

class Estimator_Step:
    def __init__(self, cfg):
        self.cfg = cfg

    def train_worker(self, gpu_id: int, world_size: int, cfg: OmegaConf):
        torch.cuda.set_device(gpu_id)
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            timeout=timedelta(seconds=720000),
            rank=gpu_id,
            world_size=world_size,
        )

        # Prepare the dataset
        register_datasets_in_cfg(cfg)
        dataset_meta = MetadataCatalog.get(cfg.datasets.train[0])
        obj_names = dataset_meta.objs

        # Load data
        train_dset_names = cfg.datasets.train
        data_loader = build_gdrn_self_train_loader(cfg, train_dset_names, train_objs=obj_names)
        data_loader_iter = iter(data_loader)

        batch_size = cfg.train.batch_size
        dataset_len = len(data_loader.dataset)
        iters_per_epoch = dataset_len // batch_size // world_size
        max_iter = cfg.train.epochs * iters_per_epoch
        dprint("batch_size: ", batch_size)
        dprint("dataset length: ", dataset_len)
        dprint("iters per epoch: ", iters_per_epoch)
        dprint("total iters: ", max_iter)

        # Initialize the model
        model, params = load_model(cfg)
        model.cuda()
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[gpu_id],
            find_unused_parameters=True,
        )             
        model.train()
        
        # Initialize the optimizer
        optimizer = build_optimizer(cfg, params)
        # Initialize the scheduler
        scheduler = solver_utils.build_lr_scheduler(cfg, optimizer, total_iters=max_iter)

        # Initialize the checkpointer
        grad_scaler = GradScaler()
        checkpointer = MyCheckpointer(
            model,
            save_dir=cfg.output_dir,
            optimizer=optimizer,
            scheduler=scheduler,
            gradscaler=grad_scaler,
            save_to_disk=comm.is_main_process(),
        )
        # Load the checkpoint for student model
        start_iter = checkpointer.resume_or_load(cfg.model.pretrained, resume=False).get("iteration", -1) + 1
        start_epoch = start_iter // iters_per_epoch + 1  # first epoch is 1

        # Initialize the DIBR renderer
        data_ref = ref.__dict__[dataset_meta.ref_key]
        ren_models, ren = get_dibr_models_renderer(cfg, data_ref, obj_names=obj_names)

        if cfg.train.checkpoint_by_epoch:
            ckpt_period = cfg.train.checkpoint_period * iters_per_epoch
        else:
            ckpt_period = cfg.train.checkpoint_period
        periodic_checkpointer = PeriodicCheckpointer(
            checkpointer, ckpt_period, max_iter=max_iter, max_to_keep=cfg.train.max_to_keep
        )
        
        # Build writers
        tbx_event_writer = get_tbx_event_writer(cfg.output_dir, backup=not cfg.get("RESUME", False))
        tbx_writer = tbx_event_writer._writer  # NOTE: we want to write some non-scalar data
        writers = (
            [MyCommonMetricPrinter(max_iter), MyJSONWriter(osp.join(cfg.output_dir, "metrics.json")), tbx_event_writer]
            if comm.is_main_process()
            else []
        )
        # test(cfg, model.module, epoch=0, iteration=0)
        logger.info("Starting training from iteration {}".format(start_iter))
        iter_time = None
        with EventStorage(start_iter) as storage:
            optimizer.zero_grad(set_to_none=True)
            for iteration in range(start_iter, max_iter):
                storage.iter = iteration
                epoch = iteration // iters_per_epoch + 1  # epoch start from 1
                storage.put_scalar("epoch", epoch, smoothing_hint=False)
                is_log_iter = False
                if iteration - start_iter > 5 and (
                    (iteration + 1) % cfg.train.print_freq == 0 or iteration == max_iter - 1 or iteration < 100
                ):
                    is_log_iter = True
                
                data = next(data_loader_iter)

                if iter_time is not None:
                    storage.put_scalar("time", time.perf_counter() - iter_time)
                iter_time = time.perf_counter()

                # Forward
                file_names = []
                batch = batch_data_self(cfg, data, file_names)

                # only outputs, no losses
                out_dict = model(
                    batch["roi_img"],
                    gt_points=batch.get("roi_points", None),
                    sym_infos=batch.get("sym_info", None),
                    roi_classes=batch["roi_cls"],
                    roi_cams=batch["roi_cam"],
                    roi_whs=batch["roi_wh"],
                    roi_centers=batch["roi_center"],
                    resize_ratios=batch["resize_ratio"],
                    roi_coord_2d=batch.get("roi_coord_2d", None),
                    roi_extents=batch.get("roi_extent", None),
                    do_self=True,
                )
                
                # Compute losses
                loss_dict = compute_self_loss(
                    cfg,
                    batch,
                    out_dict["rot"],
                    out_dict["trans"],
                    ren,
                    ren_models,
                    tb_writer=tbx_writer,
                    iteration=iteration,
                )
                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict

                loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

                # Backward
                losses.backward()
                # optimize
                # set nan grads to 0
                for param in model.parameters():
                    if param.grad is not None:
                        nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                scheduler.step()

                # Do test
                if cfg.test.eval_period > 0 and (iteration + 1) % cfg.test.eval_period == 0 and iteration != max_iter - 1:
                    self.test(cfg, model, epoch=epoch, iteration=iteration)
                    # Compared to "train_net.py", the test results are not dumped to EventStorage
                    comm.synchronize()
                
                # Logger
                if is_log_iter:
                    for writer in writers:
                        writer.write()

                # Checkpointer step
                periodic_checkpointer.step(iteration, epoch=epoch)
            
            writer.close()
            # Do test at last 
            self.test(cfg, model, epoch=epoch, iteration=iteration)
            comm.synchronize()

    @torch.no_grad()
    def test(self, cfg, model, epoch=None, iteration=None):
        results = OrderedDict()
        model_name = osp.basename(cfg.model.pretrained).split('.')[0]
        for dataset_name in cfg.datasets.test:
            if epoch is not None and iteration is not None:
                eval_out_dir = osp.join(cfg.output_dir, f'test_epoch_{epoch:02d}_iter_{iteration:06d}', dataset_name)
            else:
                eval_out_dir = osp.join(cfg.output_dir, f'test_{model_name}', dataset_name)
            evaluator = self.build_evaluator(cfg, dataset_name, eval_out_dir)
            data_loader = build_gdrn_test_loader(cfg, dataset_name, train_objs=evaluator.train_objs)
            results_i = gdrn_inference_on_dataset(cfg, model, data_loader, evaluator)
            results[dataset_name] = results_i
        if len(results) == 1:
            results = list(results.values())[0]
        return results

    def build_evaluator(self, cfg, dataset_name, output_folder=None):
        """Create evaluator(s) for a given dataset.

        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = osp.join(cfg.output_dir, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        _distributed = comm.get_world_size() > 1
        dataset_meta = MetadataCatalog.get(cfg.datasets.train[0])
        train_obj_names = dataset_meta.objs
        if evaluator_type == "bop":
            gdrn_eval_cls = GDRN_EvaluatorCustom
            return gdrn_eval_cls(
                cfg, dataset_name, distributed=_distributed, output_dir=output_folder, train_objs=train_obj_names
            )
        else:
            raise NotImplementedError(
                "Evaluator type {} is not supported for dataset {}".format(evaluator_type, dataset_name)
            )

    def cleanup(self):
        dist.destroy_process_group()

    def run(self):
        cfg = self.cfg
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
        if cfg.get('gpus') is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus
            world_size = len(cfg.gpus.split(','))
        else:
            world_size = 1

        try:
            mp.spawn(self.train_worker, nprocs=world_size, args=(world_size, cfg,))
        except Exception as e:
            print(f'Exception: {e}')
            self.cleanup()

if __name__ == "__main__":
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('configs/estimator/lm/self_train.yaml')
    cfg_parent = OmegaConf.load('configs/estimator/base.yaml')
    cfg = OmegaConf.merge(cfg_parent, cfg)
    self_train = Estimator_Step(cfg)
    self_train.run()
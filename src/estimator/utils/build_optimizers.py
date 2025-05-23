# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import torch
from mmcv import ConfigDict
from mmcv.runner.optimizer import (
    OPTIMIZERS,
    DefaultOptimizerConstructor,
    build_optimizer,
)
from mmcv.utils import build_from_cfg
from .config_utils import try_get_key
from omegaconf import OmegaConf
from enum import Enum
from .torch_utils.solver.grad_clip_d2 import maybe_add_gradient_clipping

class GradientClipType(Enum):
    VALUE = "value"
    NORM = "norm"
    FULL_MODEL = "full_model"

def register_optimizer(name):
    """TODO: add more optimizers"""
    if name in OPTIMIZERS:
        return
    if name == "Ranger":
        from .torch_utils.solver.ranger import Ranger

        # from lib.torch_utils.solver.ranger2020 import Ranger
        OPTIMIZERS.register_module()(Ranger)
    elif name == "MADGRAD":
        from .torch_utils.solver.madgrad import MADGRAD

        OPTIMIZERS.register_module()(MADGRAD)
    elif name in ["AdaBelief", "RangerAdaBelief"]:
        from .torch_utils.solver.AdaBelief import AdaBelief
        from .torch_utils.solver.ranger_adabelief import RangerAdaBelief

        OPTIMIZERS.register_module()(AdaBelief)
        OPTIMIZERS.register_module()(RangerAdaBelief)
    elif name in ["SGDP", "AdamP"]:
        from .torch_utils.solver.adamp import AdamP
        from .torch_utils.solver.sgdp import SGDP

        OPTIMIZERS.register_module()(AdamP)
        OPTIMIZERS.register_module()(SGDP)
    elif name in ["SGD_GC", "SGD_GCC"]:
        from .torch_utils.solver.sgd_gc import SGD_GC, SGD_GCC

        OPTIMIZERS.register_module()(SGD_GC)
        OPTIMIZERS.register_module()(SGD_GCC)
    else:
        raise ValueError(f"Unknown optimizer name: {name}")


def build_optimizer(config, params):
    optim_cfg = config.train.optimizer_cfg
    optim_cfg = copy.deepcopy(optim_cfg)
    register_optimizer(optim_cfg.type)

    # convert to mmcv config
    optim_cfg = OmegaConf.to_container(optim_cfg, resolve=True) 
    optim_cfg = ConfigDict(optim_cfg)
    optim_cfg['params'] = params
    optimizer = build_from_cfg(optim_cfg, OPTIMIZERS)
    return maybe_add_gradient_clipping(config, optimizer)
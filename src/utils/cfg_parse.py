import torch
import os
import random
import numpy as np
from omegaconf import OmegaConf

def cfg_parse_carf(cfg):
    # Merge parent
    if cfg.get("__parent__") is not None:
        cfg_parent = OmegaConf.load(cfg.get("__parent__"))
        cfg = OmegaConf.merge(cfg_parent, cfg)

    # Get W and H
    H = cfg.data.image_size[0]
    W = cfg.data.image_size[1]
    cfg.W, cfg.H = W, H

    # Set device
    if torch.cuda.is_available():
        cfg.device = "cuda"
    else:
        print("CUDA is not available. Using CPU.")
        cfg.device = "cpu"

    # Set output path
    cfg.output = os.path.join(cfg.output_root, 'carf', cfg.group, cfg.data.scene)
    
    # Set seed
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        
    return cfg

def cfg_parse_estimator(cfg):
    # Merge parent
    if cfg.get("__parent__") is not None:
        cfg_parent = OmegaConf.load(cfg.get("__parent__"))
        cfg = OmegaConf.merge(cfg_parent, cfg)

    # Set seed
    if cfg.get("seed") is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    return cfg
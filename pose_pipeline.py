import torch
import importlib
import argparse
from src.estimator.utils import logger
from omegaconf import OmegaConf
from src.utils.cfg_parse import cfg_parse_estimator

class pose_pipeline():
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self):
        step = self.cfg.step

        logger.info(f'Run step: {step}...')
        module = importlib.import_module(f"src.estimator.{step}")
        est_step_class = getattr(module, 'Estimator_Step')
        est_step = est_step_class(self.cfg)
        est_step.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)

    args, overrides = parser.parse_known_args()
    cli_cfg = OmegaConf.from_dotlist(overrides)
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    cfg = cfg_parse_estimator(cfg)

    pose_pipeline(cfg).run()
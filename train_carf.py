# train.py
import importlib
import logging
import os
from omegaconf import OmegaConf
from src.utils.cfg_parse import cfg_parse_carf


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self):
        base_scene = self.cfg.data.scene
        scene_path = os.path.join('splits', self.cfg.data.dataset, self.cfg.data.object)
        scene_splits = os.listdir(scene_path)
        scene_splits = [s for s in scene_splits if base_scene in s]
        scene_splits.sort()

        for scene in scene_splits[1:]:
            self.cfg.data.scene = scene
            self.cfg = cfg_parse_carf(self.cfg)
            self.run_scene()

    def run_scene(self):
        model = self.cfg['model']
        module = importlib.import_module(f"src.carf.model.{model}")
        carf_step_class = getattr(module, 'Carf_Step')
        carf_step = carf_step_class(self.cfg)
        carf_step.setup(self.cfg)

        carf_step.train(self.cfg)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    
    args, overrides = parser.parse_known_args()
    cli_cfg = OmegaConf.from_dotlist(overrides)
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    Trainer(cfg).run()
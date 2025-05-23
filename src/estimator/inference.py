import torch
import os.path as osp

from collections import OrderedDict
from detectron2.data import MetadataCatalog
from .utils import common as comm
from .utils.gdrn_custom_evaluator import GDRN_EvaluatorCustom
from .data.data_loader import build_gdrn_self_train_loader, build_gdrn_test_loader
from .utils.gdrn_evaluator import gdrn_inference_on_dataset
from .utils.load_model import load_model
from .data.dataset_factory import register_datasets_in_cfg

class Estimator_Step():
    def __init__(self, cfg):
        self.cfg = cfg
        self.setup()
    
    def setup(self):
        cfg = self.cfg

        # Build Networks
        model, _ = load_model(cfg)
        model.cuda()
        self.model = model

    def run(self):
        self.model.eval()
        self.test_model(self.cfg, self.model)
        

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

    @torch.no_grad()
    def test_model(self, config, model):
        register_datasets_in_cfg(config)
        results = OrderedDict()
        for dataset_name in config.datasets.test:
            eval_out_dir = osp.join(config.output_dir, 'test_result', dataset_name)
            evaluator = self.build_evaluator(config, dataset_name, eval_out_dir)
            data_loader = build_gdrn_test_loader(config, dataset_name, train_objs=evaluator.train_objs)
            results_i = gdrn_inference_on_dataset(config, model, data_loader, evaluator)
            results[dataset_name] = results_i
        if len(results) == 1:
            results = list(results.values())[0]
        return results


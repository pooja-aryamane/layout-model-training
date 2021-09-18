import os
import argparse
import torch
import detectron2
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.config import get_cfg
from detectron2 import model_zoo

#args init
parser = argparse.ArgumentParser()

parser.add_argument("-train_type", type=str, help="How the model should be trained? : [scratch, coco, publaynet, docbank]")
parser.add_argument("-model", type=str, help="Which model do you want to train? : [maskrcnn, fasterrcnn]")
parser.add_argument("-data", type=str, help="Which data do you want to train with? : [original, synthetic]")
parser.add_argument("-log", default="log/trainlog.log", type=str, help="Path to log file. Default [log/trainlog.log]")

args = parser.parse_args()
    
specs = args.model+"_"+args.train_type+"_"+args.data

config_specs = "configs.sanskrit-configs."+specs

m = __import__ (config_specs)
try:
    attrlist = m.__all__
except AttributeError:
    attrlist = dir (m)
for attr in attrlist:
    globals()[attr] = getattr (m, attr)
    
setup_logger(args.log)

register_coco_instances("train_data", {}, trainjson, trainimages)
register_coco_instances("test_data", {}, testjson, testimages)
register_coco_instances("val_data", {}, valjson, valimages)

class SanskritTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs(specs+"_eval", exist_ok=True)
            output_folder = specs+"_eval"
        return COCOEvaluator(dataset_name, cfg, False, output_folder)


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = SanskritTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()




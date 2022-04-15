
import logging
import os
import json
import datetime
from collections import OrderedDict
import torch
import argparse
from detectron2.utils.logger import setup_logger

import detectron2
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA


from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger

from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.data import build_detection_test_loader
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

logger = logging.getLogger("detectron2")

dataroot='./sanskrit-layout-gt-manual-old'
traindata=dataroot+'/train'
testdata=dataroot+'/test'
valdata=dataroot+'/val'
trainjson=traindata+'/train.json'
testjson=testdata+'/test.json'
valjson=valdata+'/val.json'
trainimages=traindata+'/images'
testimages=testdata+'/images'
valimages=valdata+'/images'

register_coco_instances("train_data", {}, trainjson, trainimages)
register_coco_instances("test_data", {}, testjson, testimages)
register_coco_instances("val_data", {}, valjson, valimages)

cfg = get_cfg()
config_filePath = "configs/sanskrit_configs"

index = 1
config_filesDict = {}
for cfile in os.listdir("configs/sanskrit_configs"):
    config_filesDict[index] = cfile
    print(index,":",cfile)
    index+=1

print(" ")
chosenFile = input("Select Model Config for Evaluation : ")
print("Selected Model = ",config_filesDict[int(chosenFile)])

config_file = config_filePath + '/' + config_filesDict[int(chosenFile)]
print(" ")

cfg.merge_from_file(config_file)

weightsPath = input("Enter Path of Weights: ")
evalOutput = input ("Enter Name of Eval Output Dir (to create): ")
evalData = input("Enter Data for Eval (eg. train_data, test_data, val_data): ")
cfg.MODEL.WEIGHTS = weightsPath
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
cfg.OUTPUT_DIR=evalOutput
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
evaluator = COCOEvaluator(evalData, cfg, False, output_dir=evalOutput)
test_loader = build_detection_test_loader(cfg, evalData)
model = build_model(cfg)
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)  # load a file, usually from cfg.MODEL.WEIGHTS
print(inference_on_dataset(model , test_loader, evaluator))
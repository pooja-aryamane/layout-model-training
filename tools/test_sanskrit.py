import logging
import os
import json
import pandas as pd
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

logger = logging.getLogger("detectron2")


class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
          output_folder = cfg.OUTPUT_DIR
        evaluator = COCOEvaluator(dataset_name, cfg, False, output_folder)
        val_loader = build_detection_test_loader(cfg, dataset_name)
        results=inference_on_dataset(trainer.model,val_loader,evaluator)
        pd.DataFrame(results).to_csv(f'{cfg.OUTPUT_DIR}/eval.csv')
        print_csv_format(results)
        return results



def main(args):
    print("main(args) is called")

    if args.eval_only:
      Trainer.build_evaluator(cfg, f"{args.dataset_name}_test")
    


if __name__ == "__main__":
    parser = default_argument_parser()
    #parser = argparse.ArgumentParser()

    '''here default arguments for config-file, dataset-name,eval-only(weights),testpaths,
    output-directory can be changed as desired'''
    parser.add_argument("-config-file", type=str, default = "Misc/scratch_mask_rcnn_R_50_FPN_3x_gn.yaml", metavar="FILE", help="config file")
    parser.add_argument("-dataset-name", type=str, default ="mydata", help="Dataset name")
    json_test = parser.add_argument("-json-annotation-test", type = str, metavar="FILE", help="The path to the test set JSON annotation")
    image_test = parser.add_argument("-image-path-test", type = str, metavar="FILE", help="The path to the test set image folder")
    #train folder path is required because cfg.TRAIN can't be empty, so registering test for train, if no train folder path is given
    parser.add_argument("-json-annotation-train", type = str, default = json_test, metavar="FILE", help="The path to the training set JSON annotation")
    parser.add_argument("-image-path-train", type = str, default = image_test, metavar="FILE", help="The path to the training set image folder")
    parser.add_argument("-eval-only", type = str, default = "model_final_from_scratch.pth", metavar="FILE", help="weights for evaluation")
    parser.add_argument("-output-directory", metavar="FILE",default = "/content/drive/MyDrive/TESTING", help="The path of the output folder")
    
    args = parser.parse_args()

    print("Command Line Args:", args)

    cfg = get_cfg()
  

    output_dir = args.output_directory
    #output_dir = "/content/drive/MyDrive/TESTING"
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.merge_from_file(args.config_file)
    #cfg.merge_from_list(args.opts)

    
    dataset_name = args.dataset_name
    print("Your dataset is registered as",dataset_name)

    # Register Datasets 
    '''if we wish to not have train set for inferring, but still not leave cfg.DATASETS.TRAIN empty,
    just register test set for both the coco instances'''

    if(args.json_annotation_train == json_test and args.image_path_train == image_test):
      print("Registering test set for train also, since there is no train set argument...")
      register_coco_instances(f"{dataset_name}_test", {}, args.json_annotation_test, args.image_path_test)
      register_coco_instances(f"{dataset_name}_train", {}, args.json_annotation_test, args.image_path_test)
      
      cfg.DATASETS.TEST = (f"{args.dataset_name}_test",)
      cfg.DATASETS.TRAIN = (f"{args.dataset_name}_test",)

    else:
      register_coco_instances(f"{dataset_name}_test", {}, args.json_annotation_test, args.image_path_test)
      register_coco_instances(f"{dataset_name}_train", {}, args.json_annotation_train, args.image_path_train)
      
      cfg.DATASETS.TEST = (f"{args.dataset_name}_test",)
      cfg.DATASETS.TRAIN = (f"{args.dataset_name}_train",)

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS =  args.eval_only 
    cfg.SOLVER.IMS_PER_BATCH = 10
    cfg.SOLVER.BASE_LR = 0.003  # pick a good LR
    cfg.SOLVER.MAX_ITER = 10    
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 10  #(default: 512)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    
    with open(args.json_annotation_test, 'r') as fp:
        annot_file = json.load(fp)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(annot_file["categories"])
    print("ROI Heads is taken as",cfg.MODEL.ROI_HEADS.NUM_CLASSES)

    cfg.TEST.EVAL_PERIOD = 1000
    trainer = DefaultTrainer(cfg) 
    #trainer = CocoTrainer(cfg)
    trainer.resume_or_load(resume=False)
    main(args)

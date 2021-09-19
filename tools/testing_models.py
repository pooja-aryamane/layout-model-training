import logging
import os
import datetime
from collections import OrderedDict
import torch

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
        evaluator = COCOEvaluator(dataset_name)
        val_loader = build_detection_test_loader(cfg, dataset_name)
        #print(inference_on_dataset(trainer.model,val_loader, evaluator))
        results=inference_on_dataset(trainer.model,val_loader,evaluator)
        print_csv_format(results)
        return results



def main(args):
    print("CALLED")

    if args.eval_only:
      Trainer.build_evaluator(cfg,"my_dataset_val")
    


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    output_dir = "/content/drive/MyDrive/TESTING"
    os.makedirs(output_dir, exist_ok=True)


    register_coco_instances("my_dataset_train", {}, "/content/drive/MyDrive/Layout-COCO/train_coco/train.json", "/content/drive/MyDrive/Layout-COCO/train_coco/training")
    register_coco_instances("my_dataset_val", {}, "/content/drive/MyDrive/Layout-COCO/validation_coco/val.json", "/content/drive/MyDrive/Layout-COCO/validation_coco/data")
    register_coco_instances("my_dataset_test", {}, "/content/drive/MyDrive/Layout-COCO/test_coco/test.json", "/content/drive/MyDrive/Layout-COCO/test_coco/data")
    register_coco_instances("my_dataset_val1", {}, "/content/examples/samples.json", "/content/examples/data1") 


    cfg = get_cfg()
    cfg.OUTPUT_DIR = output_dir

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)


    #cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    #cfg.merge_from_file("/content/detectron2/configs/DLA_mask_rcnn_X_101_32x8d_FPN_3x.yaml")

    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_test",)
    cfg.DATALOADER.NUM_WORKERS = 2

    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("DLA_mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    #cfg.MODEL.WEIGHTS = "/content/drive/MyDrive/model_0005499_new.pth"

    cfg.SOLVER.IMS_PER_BATCH = 10
    cfg.SOLVER.BASE_LR = 0.003  # pick a good LR
    cfg.SOLVER.MAX_ITER = 10    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 10   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    cfg.TEST.EVAL_PERIOD = 1000
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    #trainer = CocoTrainer(cfg)
    trainer.resume_or_load(resume=False)
    #trainer.train()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
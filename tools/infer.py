import sys
import requests
import tarfile
import json
import numpy as np
from os import path
from PIL import Image
from PIL import ImageFont, ImageDraw
from glob import glob
from matplotlib import pyplot as plt

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import argparse
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch

# import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode


from detectron2.structures import BoxMode

from detectron2.data.datasets import register_coco_instances

if __name__ == "__main__":
    parser = default_argument_parser()
    #parser = argparse.ArgumentParser()

    #here default arguments can be changed as desired
    parser.add_argument("-config-file", type=str, default = "/content/detectron2/configs/Misc/scratch_mask_rcnn_R_50_FPN_3x_gn.yaml", metavar="FILE", help="config file")
    parser.add_argument("-dataset-name", type=str, default ="mydata", help="Dataset name")
    #train path is got to set the thing classes
    parser.add_argument("-json-annotation-train", type = str, metavar="FILE", help="The path to the training set JSON annotation")
    parser.add_argument("-image-path-train", type = str, metavar="FILE", help="The path to the training set image folder")
    parser.add_argument("-input-image-path", type = str, default = "/content/drive/MyDrive/Layout-COCO/test_coco/data/GK2_page-0001.jpg", metavar="FILE", help="The path to the test image")
    parser.add_argument("-confidence-threshold", type = float, default = 0.5, help="threshold for predictions")
    parser.add_argument("-model-weights", type = str, metavar="FILE",default = "/content/drive/MyDrive/model_final_from_scratch.pth", help="Weights")
    parser.add_argument("-output-file", metavar="FILE",default = "/content/drive/MyDrive/RESULTS_FOR_DEMO/inference.jpg", help="The path of the output directory")
    
    
    args = parser.parse_args()

    print("Command Line Args:", args)

    cfg = get_cfg()

    filename = os.path.join(args.output_file)
    dirname = os.path.dirname(filename)
    print("DIRENAME IS",dirname)
    os.makedirs(dirname, exist_ok=True)

    cfg.merge_from_file(args.config_file)
    
    dataset_name = args.dataset_name
    print("Your dataset is registered as",dataset_name)
    register_coco_instances(f"{dataset_name}_train", {}, "/content/drive/MyDrive/Layout-COCO/train_coco/train.json", "/content/drive/MyDrive/Layout-COCO/train_coco/training")
    MetadataCatalog.get(f"{dataset_name}_train").set(thing_classes=["bg","Image","Math","Table","Text"])
    layout_metadata = MetadataCatalog.get(f"{dataset_name}_train")
    print("Metadata is",layout_metadata)

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold # set threshold for this model
    
    with open(args.json_annotation_train, 'r') as fp:
        annot_file = json.load(fp)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(annot_file["categories"])
    print("ROI Heads is taken as",cfg.MODEL.ROI_HEADS.NUM_CLASSES)
    
    cfg.MODEL.WEIGHTS =  args.model_weights
    predictor = DefaultPredictor(cfg)

    im = cv2.imread(args.input_image_path)
    cv2_imshow(im)
    outputs = predictor(im)
    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)  

    v = Visualizer(im[:, :, ::-1],
                      metadata=layout_metadata, 
                      scale=0.5, 
                      instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
    print(layout_metadata)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    ans = out.get_image()[:, :, ::-1]
    cv2_imshow(out.get_image()[:, :, ::-1])
    im = Image.fromarray(ans)
    print(filename)
    #im.save(filename)
    cv2.imwrite(filename,out.get_image()[:, :, ::-1])
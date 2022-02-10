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
import sys
sys.path.insert(0,'configs/sanskrit-configs')


#args init
parser = argparse.ArgumentParser()

parser.add_argument("-train_type", type=str, help="How the model should be trained? : [scratch, coco, publaynet, docbank]")
parser.add_argument("-model", type=str, help="Which model do you want to train? : [maskrcnn, fasterrcnn]")
parser.add_argument("-data", type=str, help="Which data do you want to train with? : [original, synthetic, combined]")
parser.add_argument("-log", default="log/trainlog.log", type=str, help="Path to log file. Default [log/trainlog.log]")
parser.add_argument("-lr",default=None, type=float, help="learning rate for training eg: 0.0001")
parser.add_argument("-max_iter", default=None, type=int, help="maximum number of training iterations eg: 20000")
parser.add_argument("-batch_size", default=None, type=int, help="batch size per image eg: 64 or 512")
parser.add_argument("-output_folder", default=None, type=str, help="name of the output folder")


args = parser.parse_args()

setup_logger(args.log)

specs = args.model+"_"+args.train_type

print("specs : ",specs)
#os.system('python'+specs+'.py')

dataroot='./sanskrit-layout-gt-manual-old'
traindata=dataroot+'/train'
testdata=dataroot+'/test'
valdata=dataroot+'/val'
if args.data == "original":
    trainjson=traindata+'/train.json'
elif args.data == "combined":
    trainjson=traindata+'/orig_synth.json'
elif args.data == "synthetic":
    trainjson = dataroot + '/synthetic_coco/synth_labels.json'
testjson=testdata+'/test.json'
valjson=valdata+'/val.json'
if args.data == "synthetic":
    trainimages = dataroot + '/synthetic_coco/images'
else:
    trainimages=traindata+'/images'
testimages=testdata+'/images'
valimages=valdata+'/images'

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

#load the config
cfg = get_cfg()
cfg.merge_from_file("configs/sanskrit_configs/maskrcnn_publaynet.yaml")
cfg.DATASETS.TRAIN=('train_data',)
cfg.DATASETS.TEST=('val_data',)

# lr,iterations,batch_size
if args.lr is not None:
  cfg.SOLVER.BASE_LR = args.lr
if args.max_iter is not None:
  cfg.SOLVER.MAX_ITER = args.max_iter
if args.batch_size is not None:
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.batch_size
if args.output_folder is not None:
  cfg.OUTPUT_DIR = args.output_folder
else:
  cfg.OUTPUT_DIR = specs+"_output"

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = SanskritTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()




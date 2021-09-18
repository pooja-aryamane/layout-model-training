from detectron2.config import get_cfg
from detectron2 import model_zoo

#datainitializations
dataroot='./data/sanskrit'
traindata=dataroot+'/train'
testdata=dataroot+'/test'
valdata=dataroot+'/val'
trainjson=traindata+'/train.json'
testjson=testdata+'/test.json'
valjson=valdata+'/val.json'
trainimages=traindata+'/images'
testimages=testdata+'/images'
valimages=valdata+'/images'

#yaml config init
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("Misc/scratch_mask_rcnn_R_50_FPN_3x_gn.yaml"))
cfg.DATASETS.TRAIN = ("train_data",)
cfg.DATASETS.TEST = ("val_data",)

cfg.DATALOADER.NUM_WORKERS = 2

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Misc/scratch_mask_rcnn_R_50_FPN_3x_gn.yaml")  # initialize from scratch

cfg.SOLVER.IMS_PER_BATCH = 2 
cfg.SOLVER.BASE_LR = 0.001

cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 20000 #adjust up if val mAP is still rising, adjust down if overfit
cfg.SOLVER.STEPS = [] 
# cfg.SOLVER.STEPS = [] #if we don't want to decay learning
cfg.SOLVER.GAMMA = 0.05

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5 #your number of classes + 1

cfg.TEST.EVAL_PERIOD = 500
cfg.OUTPUT_DIR= "maskrcnn_scratch_original_output"
cfg.SOLVER.CHECKPOINT_PERIOD = 2000


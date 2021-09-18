from detectron2.config import get_cfg

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
cfg.merge_from_file("DLA_mask_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("train_data",)
cfg.DATASETS.TEST = ("test_data",)

cfg.DATALOADER.NUM_WORKERS = 1
cfg.MODEL.WEIGHTS = ("/home/server/Documents/layout/layoutanalysis/diff_models/model_maskrcnn_publaynet_finetuned.pth")  # initialize from scratch
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.001

cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 20000 #adjust up if val mAP is still rising, adjust down if overfit
cfg.SOLVER.STEPS = (100, 150)
# cfg.SOLVER.STEPS = [] #if we don't want to decay learning
cfg.SOLVER.GAMMA = 0.05

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5 #your number of classes + 1

cfg.TEST.EVAL_PERIOD = 500
cfg.OUTPUT_DIR= "maskrcnn_publaynet_original_output"
cfg.SOLVER.CHECKPOINT_PERIOD = 2000

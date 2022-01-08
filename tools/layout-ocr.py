import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog

import json
import os
import detectron2
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.data.datasets import register_coco_instances
#We are importing our own Trainer Module here to use the COCO validation evaluation during training. Otherwise no validation eval occurs.

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

try:
 from PIL import Image
except ImportError:
 import Image
import cv2
import pytesseract
import os
from pdf2image import convert_from_path
import cv2
import sys
from pdfreader import SimplePDFViewer
import shutil

#args init
parser = argparse.ArgumentParser()

parser.add_argument("-model_index", type=str, help="Which model do you want to use to detect document layout? : Please choose 1,2,3,4,5 or 6 where -  [1:fasterrcnn_publaynet, 2:fasterrcnn_coco, 3:fasterrcnn_scratch, 4:maskrcnn_publaynet, 5:maskrcnn_coco, 6:maskrcnn_scratch]")
parser.add_argument("-ocr_lang", type=str, help="Which language model do you want to use? : [1:san, 2:san_iitb]")
parser.add_argument("-img_dir", type=str, help="Image directory")

args = parser.parse_args()

img_dir = args.img_dir
layout_model_num = args.model_index
ocr_language = args.ocr_lang

#config folder should contain all configs
#model folder should contain all model weights

os.chdir("config")

model_indices = {1:'fasterrcnn_publaynet', 2:'fasterrcnn_coco', 3:'fasterrcnn_scratch',
                 4:'maskrcnn_publaynet', 5:'maskrcnn_coco', 6:'maskrcnn_scratch' }

languages = {1:'san', 2:'san_iitb'}

config = 'config/' + model_indices[layout_model_num] + '.yaml'
model = model_indices[layout_model_num] + '.pth'
cfg = get_cfg()
cfg.merge_from_file(config)

#########cfgs###################################
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = 'model/'+model
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 1500 #adjust up if val mAP is still rising, adjust down if overfit
cfg.SOLVER.STEPS = (1000, 1500)
cfg.SOLVER.GAMMA = 0.05
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5 #your number of classes + 1
cfg.TEST.EVAL_PERIOD = 500
cfg.SOLVER.CHECKPOINT_PERIOD = 500
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
#########cfgs###################################
    
predictor = DefaultPredictor(cfg)
test_metadata = MetadataCatalog.get("val")

os.mkdir("output")

for img in os.listdir(img_dir):
    image_path = img_dir + "/" + img
    os.mkdir("/output/" + img)
    MetadataCatalog.get("val").thing_classes = ["bg", "Image","Math","Table","Text"]
    if os.path.isfile(image_path):
        im = cv2.imread(image_path)
        image = Image.open(image_path)
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],metadata = MetadataCatalog.get("val") , scale=0.8)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        #cv2_imshow(out.get_image()[:, :, ::-1])
        c=0
        i=0
        ls=[]
        for i in outputs["instances"].pred_classes:
          k = int(i.tolist())
          if (k==4):
            temp = outputs["instances"].pred_boxes[c]
            row=temp.tensor.tolist()[0]
            ls.append(row)
          c+=1

        ls=sorted(ls, key=itemgetter(1))

        for row in ls:
          i+=1
          crop_box = (row[0], row[1], row[2], row[3])
          crop_box = map(int, crop_box)
          test = image.crop((crop_box))
          test.show()
          path = "output/" + img + "/text" + str(i.tolist()) + ".jpeg"
          test.save(path)

#Tesseract OCR

tessdata_dir_config = r'--tessdata-dir "/path/to/tessdata/"'


os.environ['CHOSENMODEL']=languages[ocr_language]

for filename in os.listdir('output/'):
#CHANGE IMAGESFOLDER FOR BATCHES
os.environ['IMAGESFOLDER']='output/'+filename+'' #CHANGE HERE
os.environ['OUTPUTDIRECTORY']='ocr_output'
os.environ['CHOSENFILENAMEWITHNOEXT']='ocr'

!find $IMAGESFOLDER -maxdepth 1 -type f > $OUTPUTDIRECTORY/$CHOSENFILENAMEWITHNOEXT.list

os.environ['CHOSENMODEL']='san_iitb'
os.chdir('output/'+filename)

!tesseract $OUTPUTDIRECTORY/$CHOSENFILENAMEWITHNOEXT.list filename -l $CHOSENMODEL --oem 1 --psm 7 --tessdata-dir /path/to/tessdata/
print('Done '+filename)

for filename1 in os.listdir('output/'):
for filename in os.listdir('output/'+filename1):
  if filename.endswith('.txt'):
    filename2 = filename1[:-5] + '.txt'
    shutil.copy('output/'+filename1+'/'+filename , 'ocr_output/'+filename2)

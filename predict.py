#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dipu
"""


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import cv2
import pickle
import PIL
from matplotlib import pyplot as plt

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from roi_head_comat import *

dataset = 'rico'  # 'dui'
# dataset = 'dui'

metadata = pickle.load(open(f'data/{dataset}_metadata.pkl', 'rb'))
classes = metadata.thing_classes


def my_imshow(a):
    a = a.clip(0, 255).astype('uint8')
      # cv2 stores colors as BGR; convert to RGB
    if a.ndim == 3:
        if a.shape[2] == 4:
          a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
          a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    a = PIL.Image.fromarray(a)
    return(a)

def add_comat_config(cfg):
    _C = cfg
    _C.MODEL.HEAD_TYPE = ''
    _C.comat_file = ''
    _C.sgrn_gfeat = ''
    _C.use_reln_learner= ''


model_dir = f'trained_models/{dataset}_gaussian_comat_fpn/'
cfg = get_cfg()
add_comat_config(cfg)
cfg.merge_from_file(model_dir + 'config.yaml') 
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_dir + "model_final.pth"
cfg.comat_file = f'data/co_mats_10_graphs_centriod_{dataset}.pkl'

predictor = DefaultPredictor(cfg)


#%% Test on one image
imfn = 'test_images/rico/30295.jpg' if dataset == 'rico' else 'test_images/dui/6f1d84cbc371dade.jpg'
im = cv2.imread(imfn)
outputs = predictor(im)
out_labels = outputs["instances"].pred_classes.cpu().numpy()
out_bbox = outputs["instances"].pred_boxes.tensor.cpu().numpy()
out_classnames = [classes[x] for x in out_labels]
print('class_labels: ', out_labels)
print('class_names: ', out_classnames)
print('bboxes: ', out_bbox)

# Visualization 
v = Visualizer(im[:, :, ::-1],metadata, scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
img = v.get_image()[:, :, ::-1]
img = my_imshow(img)

fig, ax = plt.subplots()
plt.imshow(img)
plt.axis('off')
plt.margins(0,0)
ax.xaxis.set_major_locator(plt.NullLocator())
ax.yaxis.set_major_locator(plt.NullLocator())
plt.tight_layout()
plt.savefig('predicted.png', dpi = 300)    
    
   
    
    
  
#!/usr/bin/env python

import os
# Use cpu, we won't be doing much computation anyway, and this way it is
# trivial to run this program while training happens in the background.
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import cv2
import keras.backend as K
from keras.models import load_model
from keras.preprocessing import image
import glob
import numpy as np
from segmenty.process import process_images
import sys

def balanced_loss(y_pred, y_true):
    # dummy
    return y_pred - y_true


model = load_model(sys.argv[1], custom_objects={
    'balanced_loss': balanced_loss,
    'tpr': balanced_loss,
    'tnr': balanced_loss
})

in_dir = sys.argv[2]
out_dir = sys.argv[3]

fnames = glob.glob("{}/*.jpg".format(in_dir))
fnames = [fname for fname in fnames if '_o_' not in fname][:10]

process_images(model, fnames, out_dir, "")

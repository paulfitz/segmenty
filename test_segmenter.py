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
import numpy
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

fnames = glob.glob("{}/*_i.jpg".format(in_dir))[0:10]

def get(fname):
    img = image.load_img(fname, target_size=(256, 256))
    x = image.img_to_array(img)
    return x

CT = 0

def generate_image(x, result):
    global CT
    x = x[...,::-1]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite('%s/%06d_in.jpg' % (out_dir, CT), x)
    for i in range(0, result.shape[-1]):
        cv2.imwrite('%s/%06d_out%d.jpg' % (out_dir, CT, i),
                    result[:, :, i] * 255.0)
    CT += 1


x = numpy.stack([get(fname) for fname in fnames], axis=0)

results = model.predict(x, batch_size=20)

for fname, result in zip(x, results):
    generate_image(fname, result)


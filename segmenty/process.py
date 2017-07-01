import cv2
import keras.backend as K
from keras.models import load_model
from keras.preprocessing import image
import glob
import numpy as np
import os
import sys

def process_images(model, fnames, out_dir, prefix):

    target_size = [int(d) for d in model.inputs[0].shape[1:3]]

    def get(fname):
        img = image.load_img(fname, target_size=target_size)
        x = image.img_to_array(img)
        return x

    count = [0]

    def generate_image(x, result):
        x = x[...,::-1]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        cv2.imwrite('%s/%06d_%s_in.jpg' % (out_dir, count[0], prefix), x)
        for i in range(0, result.shape[-1]):
            cv2.imwrite('%s/%06d_%s_out%d.jpg' % (out_dir, count[0], prefix, i),
                        result[:, :, i] * 255.0)
        count[0] += 1


    x = np.stack([get(fname) for fname in fnames], axis=0)

    results = model.predict(x, batch_size=20)

    for fname, result in zip(x, results):
        generate_image(fname, result)


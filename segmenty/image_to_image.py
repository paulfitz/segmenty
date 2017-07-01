import json
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import random


class ImageToImage(object):
    def __init__(self, datafile, image_size, batch_size,
                 input_label='x', output_label='y'):
        with open(datafile, 'r') as fin:
            self.index = json.load(fin)
            self.image_size = image_size
            self.batch_size = batch_size
            self.steps = len(self.index) // batch_size
            self.input_label = input_label
            self.output_label = output_label
            self.example = self.index[0]
        self.root = '.'

    def generator(self):
        while True:
            idx = 0
            random.shuffle(self.index)
            for i in range(0, self.steps):
                xs = []
                ys = []
                for k in range(0, self.batch_size):
                    sample = self.index[idx]
                    idx = (idx + 1) % len(self.index)
                    retries = 0
                    while True:
                        try:
                            fname = os.path.join(self.root,
                                                 sample[self.input_label])
                            x_img = load_img(fname,
                                             grayscale=False,
                                             target_size=self.image_size)
                            x = img_to_array(x_img)
                            output_fnames = sample[self.output_label]
                            yy = np.zeros(self.image_size +
                                          (len(output_fnames),))
                            for i, fname in enumerate(output_fnames):
                                y_img = load_img(os.path.join(self.root,
                                                              fname),
                                                 grayscale=True,
                                                 target_size=self.image_size)
                                y = img_to_array(y_img) / 255.0
                                yy[:, :, i] = y[:, :, 0]
                            xs.append(x)
                            ys.append(yy)
                            break
                        except (IOError, TypeError):
                            retries += 1
                            if retries > 5:
                                raise
                            import time
                            time.sleep(2)
                        
                yield np.array(xs), np.array(ys)



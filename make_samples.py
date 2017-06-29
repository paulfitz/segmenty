#!/usr/bin/env python

import cv2
from glob import glob
import json
import math
import numpy as np
import os
import random
from random import randint
import sys

out = sys.argv[1]
count = int(sys.argv[2])

lst = glob('background/*/*.jpg')
index = [{'filename': fname} for fname in lst]

random.shuffle(index)
top = len(index)

W = 256
H = 256

distractors = True

samples = []


base = '.'

if not os.path.exists(out):
    os.makedirs(out)

    
def get_image(stanza):
    path = os.path.join(base, stanza['filename'])
    result = cv2.imread(path)
    return result


def shrink(img):
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)


pattern = get_image({'filename': 'pattern/full.png'})


output_patterns = []

output_names = ['mask', 'vertical', 'horizontal', 'middle']

for name in output_names:
    output_patterns.append(shrink(get_image({'filename':
                                             'pattern/{}.png'.format(name)})))

PW, PH, _ = pattern.shape


def generate_block():
    x = np.ones((PW, PH, 3))
    x[:, :, 0:3] = np.random.uniform(0, 1, (3,))

    y = np.ones((PW, PH, 3))
    y[:, :, 0:3] = np.random.uniform(0, 1, (3,))

    c = np.linspace(0, 1, PW)[:, None, None]
    gradient = x + (y - x) * c
    return x, y, gradient


def motion_blur(img):
    size = random.randint(3, 15)
    # generating the kernel
    kernel_motion_blur = np.zeros((size, size))
    x0 = int((size-1)/2)
    y0 = int((size-1)/2)
    dx = 0
    dy = 0
    while dx == 0 and dy == 0:
        dx = random.randint(-1, 1)
        dy = random.randint(-1, 1)
    ct = 0
    for k in range(-size, size):
        x = x0 + k * dx
        y = y0 + k * dy
        if x >= 0 and y >= 0 and x < size and y < size:
            kernel_motion_blur[x, y] = 1
            ct += 1
    kernel_motion_blur = kernel_motion_blur / ct
    # applying the kernel to the input image
    output = cv2.filter2D(img, -1, kernel_motion_blur)
    return output

try:

    for k in range(0, count):
        i1 = index[random.randint(0, top - 1)]
        i2 = index[random.randint(0, top - 1)]

        img1 = get_image(i1)
        img3 = get_image(i2)
        img2 = pattern  # get_image(i2)
        scale = generate_block()[2]
        img2 = np.multiply(img2, scale).astype(np.uint8)

        img1 = cv2.resize(img1, (W, H), interpolation=cv2.INTER_CUBIC)
        img2 = cv2.resize(img2, (W, H), interpolation=cv2.INTER_CUBIC)
        img3 = cv2.resize(img3, (W, H), interpolation=cv2.INTER_CUBIC)
        img1 = img1.astype(np.float) * np.random.uniform(0.33, 1.0)
        img2 = img2.astype(np.float) * np.random.uniform(0.33, 1.0)
        img3 = img3.astype(np.float) * random.uniform(0.33, 1.0)

        pts1 = np.float32([[0, 0], [W-1, 0], [W-1, H-1], [0, W-1]])
        w = W
        h = H

        x0 = randint(0, w)
        y0 = randint(0, h)
        r = np.random.normal(w // 2, w // 2)
        r = max(r, w // 8)
        r = min(r, w)

        angles = []
        angle = np.random.uniform(0, math.pi * 2)
        for _ in range(0, 4):
            if angle > math.pi * 2:
                angle -= math.pi * 2
            angles.append(angle)
            angle += np.random.uniform(math.pi / 8, math.pi * 0.75)
        angles.sort()
        angles = np.float32(angles)
        x1 = x0 + np.cos(angles) * r
        y1 = y0 + np.sin(angles) * r
        xx = np.array((x1, y1)).transpose()

        blank = (np.random.random() < 0.2)

        pts2 = np.float32(xx)
        M = cv2.getPerspectiveTransform(pts1, pts2)
        if not blank:
            cv2.warpPerspective(img2, M, (W, H), img1,
                                borderMode=cv2.BORDER_TRANSPARENT)
        delta = np.random.uniform(0.0, 0.5)
        img1 = ((1 - delta) * img1 + delta * img3).astype(np.uint8)
        dst = img1

        white_image = np.zeros(dst.shape, np.uint8)
        white_image[:, :, :] = 255

        output_images = [np.zeros(dst.shape, np.uint8) for _ in output_names]
        if not blank:
            for i in range(len(output_images)):
                cv2.warpPerspective(output_patterns[i], M, (W, H),
                                    output_images[i],
                                    borderMode=cv2.BORDER_TRANSPARENT)

        if distractors:
            for i in range(0, 10):
                cv2.ellipse(dst,
                            (random.randint(0, W-1), random.randint(0, H-1)),
                            (random.randint(0, W-1) / 5,
                             random.randint(0, H-1) / 5),
                            random.randint(0, 359),
                            0,
                            360,
                            (random.randint(0, 255),
                             random.randint(0, 255),
                             random.randint(0, 255)),
                            thickness=-1)

        width = dst.shape[1]
        height = dst.shape[0]
        distCoeff = np.zeros((4, 1), np.float64)

        k1 = random.gauss(0, 0.001)
        k2 = 0.0
        p1 = 0.0
        p2 = 0.0

        distCoeff[0, 0] = k1
        distCoeff[1, 0] = k2
        distCoeff[2, 0] = p1
        distCoeff[3, 0] = p2

        cam = np.eye(3, dtype=np.float32)

        cam[0, 2] = width / 2.0   # define center x
        cam[1, 2] = height / 2.0  # define center y
        cam[0, 0] = 10.0          # define focal length x
        cam[1, 1] = 10.0          # define focal length y

        cam[0, 2] = np.random.uniform(width * 0.4, width * 0.6)
        cam[1, 2] = np.random.uniform(height * 0.4, height * 0.6)
        cam[0, 0] = cam[1, 1] = np.random.uniform(10.0, 20.0)

        dst = cv2.undistort(dst, cam, distCoeff)
        output_images = [cv2.undistort(img, cam, distCoeff)
                         for img in output_images]

        input_fname = "%s/%06d_i.jpg" % (out, k)
        output_fnames = ["%s/%06d_o_%d_%s.png" % (out, k, idx,
                                                  name[:3])
                         for idx, name in enumerate(output_names)]
        if np.random.random() < 0.5:
            dst = motion_blur(dst)
        cv2.imwrite(input_fname, dst)
        for img, fname in zip(output_images, output_fnames):
            cv2.imwrite(fname, img)
        part = {
            'x': input_fname,
            'y': output_fnames,
            'n': k
        }
        samples.append(part)
        print part

finally:
    with open('{}.json'.format(out), 'w') as fout:
        json.dump(samples, fout, indent=2)

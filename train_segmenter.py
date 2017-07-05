#!/usr/bin/env python

from keras.models import Model
from keras.layers import Conv2D, Dropout, Input
from segmenty.hourglass import downsamples, mix, upsamples
from segmenty.run import Run

flags = {
    'image_size': (256, 256),
    'batch_size': 25
}


def model(flags):
    
    inputs = Input(shape=(flags['image_size'] + (3,)))
    x = inputs

    x, sources = downsamples(x, [40, 40, 80, 100, 100, 100, 80, 80])
    x = mix(x, 100)
    x = Dropout(0.1)(x)
    x = upsamples(x, sources, [100] * 8)

    x = Conv2D(30, (1, 1), padding='valid', activation='relu')(x)
    x = Conv2D(30, (1, 1), padding='valid', activation='relu')(x)
    x = Conv2D(len(flags['example']['y']), (1, 1), padding='valid',
               activation='sigmoid')(x)

    mod = Model(inputs=inputs, outputs=x)

    return mod


run = Run()
run.set_flags(flags)
run.use_model(model)
run.compile()
run.train()
